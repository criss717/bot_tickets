"""AI Vision API extractor for ticket data.

Supports:
1. Google Gemini (via OpenAI-compatible API)
2. Local Ollama (qwen2.5vl:3b, minicpm-v, etc.)
3. OpenCode Go / Remote OpenAI-compatible backends
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from openai import APIConnectionError, OpenAI

logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Raised when local Ollama server cannot be reached."""


@dataclass
class ExtractedTicket:
    """Data extracted from a ticket image."""

    empresa: str
    nif: str
    descripcion: str
    total: float
    fecha: str
    es_ticket: bool = True


EXTRACTION_PROMPT = """Analiza la imagen y genera un JSON con esta estructura exacta:
{
  "tipo_documento": "ticket" o "no_ticket",
  "es_ticket": true o false,
  "empresa": "nombre del comercio o vacio si no es ticket",
  "nif": "CIF/NIF o vacio",
  "descripcion": "resumen breve de lo comprado o vacio",
  "total": 0.0,
  "fecha": "DD/MM/YYYY o vacio"
}

Reglas:
1. Si la imagen es un teclado, persona, pantalla de ordenador, oficina, captura de chat/código u objeto sin gasto, pon tipo_documento: "no_ticket", es_ticket: false y total: 0.0.
2. Si es un ticket, factura, albarán o recibo de compra/gasto real, pon tipo_documento: "ticket", es_ticket: true y extrae los datos reales impresos.
3. PROHIBIDO inventar datos. Si no ves claramente un ticket o el importe final a pagar, pon total: 0.0 y es_ticket: false.
4. Responde SOLO el bloque JSON sin explicaciones."""


# ── Sanitization & Normalization Helpers ─────────────────────────────────────

SPANISH_MONTHS = {
    "enero": 1, "ene": 1,
    "febrero": 2, "feb": 2,
    "marzo": 3, "mar": 3,
    "abril": 4, "abr": 4,
    "mayo": 5, "may": 5,
    "junio": 6, "jun": 6,
    "julio": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "septiembre": 9, "sep": 9, "sept": 9,
    "octubre": 10, "oct": 10,
    "noviembre": 11, "nov": 11,
    "diciembre": 12, "dic": 12,
}

DATE_PLACEHOLDERS = {
    "", "DD/MM/YYYY", "DD-MM-YYYY", "YYYY-MM-DD", "YYYY/MM/DD",
    "DD/MM/YY", "FECHA", "HOY", "TODAY", "NULL", "NONE",
}


def clean_total(val: Any) -> float:
    """Sanitize and coerce total amount to float rounded to 2 decimals."""
    if val is None:
        return 0.0

    if isinstance(val, (int, float)):
        return max(0.0, round(float(val), 2))

    s = str(val).strip()
    if not s:
        return 0.0

    # Remove currency symbols and non-numeric chars except separators
    s = re.sub(r"[€$£EUReur\s]", "", s)
    # Replace apostrophe (common in handwritten Spanish totals like 24'00)
    s = s.replace("'", ".")

    # Handle thousands and decimal separators
    if "." in s and "," in s:
        # e.g. "1.234,56" or "1,234.56"
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        # e.g. "24,40"
        s = s.replace(",", ".")

    try:
        parsed = float(s)
        return max(0.0, round(parsed, 2))
    except (ValueError, TypeError):
        return 0.0


def clean_fecha(val: Any) -> str:
    """Sanitize, parse, and normalize date to DD/MM/YYYY.

    Falls back to current date if missing, unparseable, or placeholder.
    """
    now = datetime.now()
    default_date = now.strftime("%d/%m/%Y")

    if val is None:
        return default_date

    s = str(val).strip()
    if s.upper() in DATE_PLACEHOLDERS:
        return default_date

    # Check for Spanish text month (e.g., "21 de Mayo de 2024" or "21 mayo 2024")
    text_month_match = re.search(
        r"(\d{1,2})\s+(?:de\s+)?([a-zA-ZáéíóúÁÉÍÓÚ]+)\s+(?:de\s+)?(\d{2,4})",
        s,
        re.IGNORECASE,
    )
    if text_month_match:
        day_str, month_name, year_str = text_month_match.groups()
        month_key = month_name.lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        if month_key in SPANISH_MONTHS:
            try:
                day = int(day_str)
                month = SPANISH_MONTHS[month_key]
                year = int(year_str)
                if year < 100:
                    year += 2000
                if 2000 <= year <= now.year + 1:
                    dt = datetime(year, month, day)
                    return dt.strftime("%d/%m/%Y")
            except (ValueError, OverflowError):
                pass

    # Numeric patterns: DD/MM/YYYY or YYYY-MM-DD
    # 1. Try standard formats
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%y", "%d-%m-%y"):
        try:
            dt = datetime.strptime(s, fmt)
            if 2000 <= dt.year <= now.year + 1:
                return dt.strftime("%d/%m/%Y")
        except ValueError:
            continue

    # 2. Try regex extraction of numbers
    match = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", s)
    if match:
        d_str, m_str, y_str = match.groups()
        try:
            day, month = int(d_str), int(m_str)
            year = int(y_str)
            if year < 100:
                year += 2000
            if 2000 <= year <= now.year + 1:
                dt = datetime(year, month, day)
                return dt.strftime("%d/%m/%Y")
        except (ValueError, OverflowError):
            pass

    return default_date


def clean_nif(val: Any) -> str:
    """Sanitize and validate Spanish NIF/CIF/NIE.

    Returns uppercase alphanumeric string without dots or hyphens.
    Returns empty string if purely numeric (missing letter) or invalid.
    """
    if val is None:
        return ""

    cleaned = re.sub(r"[^A-Z0-9]", "", str(val).upper().strip())
    if not cleaned:
        return ""

    # Must contain at least one letter and at least one digit
    has_letter = bool(re.search(r"[A-Z]", cleaned))
    has_digit = bool(re.search(r"\d", cleaned))

    if not has_letter or not has_digit:
        return ""

    # Plausible Spanish format length: typically 9 characters (or 8-10)
    if len(cleaned) < 8 or len(cleaned) > 10:
        return ""

    return cleaned


def clean_empresa(val: Any) -> str:
    """Sanitize empresa string."""
    if val is None:
        return "Desconocido"
    s = str(val).strip().strip("'\"")
    # Collapse multiple whitespaces
    s = re.sub(r"\s+", " ", s)
    if not s or s.upper() in ("DESCONOCIDO", "UNKNOWN", "NONE", "NULL"):
        return "Desconocido"
    return s[:100]


def clean_descripcion(val: Any) -> str:
    """Sanitize descripcion string."""
    if val is None:
        return "Ticket"
    s = str(val).strip().strip("'\"")
    if "\n" in s or len(s) > 100 or not s:
        return "Ticket"
    return re.sub(r"\s+", " ", s)


def is_suspicious_total(total: float, threshold: float = 1000.0) -> bool:
    """Flag totals that are unusually high or zero/negative."""
    return total <= 0.0 or total >= threshold


def is_valid_ticket(ticket: ExtractedTicket) -> bool:
    """Check if extracted data corresponds to a real, valid ticket.

    Must be classified as a ticket AND have a positive monetary total.
    """
    return bool(ticket.es_ticket and ticket.total > 0.0)


# ── VisionExtractor Implementation ──────────────────────────────────────────

def _build_text_ocr_prompt(ocr_text: str) -> str:
    """Build structured JSON extraction prompt from plain OCR text."""
    return (
        f"Texto extraído por OCR de un documento:\n\"\"\"\n{ocr_text[:2000]}\n\"\"\"\n\n"
        + EXTRACTION_PROMPT.replace("Analiza la imagen", "Analiza el texto OCR anterior")
        + "\n5. Si aparece nombre comercial de bar, restaurante, tienda, hotel o empresa, ponlo en empresa."
        + "\n6. Si aparece un NIF/CIF/DNI en cualquier línea, ponlo en nif."
    )


class VisionExtractor:
    """Extracts ticket data using Vision API (Gemini, Ollama local, or OpenCode Go)."""

    GEMINI_FALLBACK_MODELS = [
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3-flash-preview",
        "gemini-flash-latest",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
    ]
    GEMINI_COOLDOWN_SECONDS = 300.0
    OLLAMA_OCR_FALLBACK_MODELS = ["glm-ocr:latest", "glm-ocr"]
    REMOTE_FALLBACK_MODELS = ["kimi-k2.6", "kimi-k2.7-code"]

    def __init__(
        self,
        api_key: str = "",
        model: str = "kimi-k2.6",
        base_url: str = "https://opencode.ai/zen/go/v1",
        is_local: bool = False,
        ollama_model: str = "qwen2.5vl:3b",
        ollama_url: str = "http://localhost:11434/v1",
        use_gemini: bool = False,
        gemini_api_key: str = "",
        gemini_api_keys: tuple[str, ...] | list[str] = (),
        gemini_model: str = "gemini-3.5-flash-lite",
    ) -> None:
        self._use_gemini = use_gemini
        self._is_local = is_local
        self._ollama_model = ollama_model
        self._ollama_url = ollama_url
        self._lock: asyncio.Lock | None = None
        self._ollama_client: OpenAI | None = None
        self._gemini_clients: list[OpenAI] = []
        self._gemini_cooldowns: dict[tuple[int, str], float] = {}
        self._disabled_gemini_pairs: set[tuple[int, str]] = set()

        if use_gemini:
            self._backend = "gemini"
            self._model = gemini_model
            keys = [k.strip() for k in gemini_api_keys if k and k.strip()]
            if gemini_api_key and gemini_api_key.strip() and gemini_api_key.strip() not in keys:
                keys.insert(0, gemini_api_key.strip())
            if not keys:
                keys = [""]

            self._gemini_clients = [
                OpenAI(
                    api_key=k,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    timeout=15.0,
                    max_retries=0,
                )
                for k in keys
            ]
            self._client = self._gemini_clients[0]
            if is_local:
                self._ollama_client = OpenAI(
                    api_key="ollama",
                    base_url=ollama_url,
                    timeout=120.0,
                    max_retries=0,
                )
            logger.info(f"Using Google Gemini: {gemini_model} ({len(self._gemini_clients)} key(s) in pool)")
        elif is_local:
            self._backend = "ollama"
            self._model = ollama_model
            self._client = OpenAI(
                api_key="ollama",
                base_url=ollama_url,
                timeout=120.0,
                max_retries=0,
            )
            self._ollama_client = self._client
            logger.info(f"Using local Ollama: {ollama_model} @ {ollama_url}")
        else:
            self._backend = "remote"
            self._model = model
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={"x-opencode-session": "bot-tickets"},
                timeout=60.0,
            )
            logger.info(f"Using remote API: {model} @ {base_url}")

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Return True if exception indicates an unreachable server/socket."""
        if isinstance(exc, APIConnectionError):
            return True
        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in ("connection error", "connection refused", "winerror 10061", "failed to establish a new connection")
        )

    @staticmethod
    def _fetch_ollama_tags(ollama_url: str = "http://localhost:11434/v1", timeout: float = 2.0) -> list[str] | None:
        """Query Ollama /api/tags; returns installed model names or None if unreachable."""
        try:
            base_url = ollama_url.rstrip("/").removesuffix("/v1")
            req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return None
                payload = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in payload.get("models", []) if isinstance(m, dict) and "name" in m]
        except Exception:
            return None

    @classmethod
    def is_ollama_available(cls, ollama_url: str = "http://localhost:11434/v1") -> bool:
        """Return True if local Ollama server responds."""
        return cls._fetch_ollama_tags(ollama_url) is not None

    def _call_model(
        self,
        model: str,
        b64_image: str,
        use_json_format: bool = True,
        client: OpenAI | None = None,
        backend: str | None = None,
    ) -> str:
        """Call a single model with the image and appropriate parameters."""
        active_client = client or self._client
        active_backend = backend or self._backend
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ]

        if active_backend == "ollama":
            # For local Ollama: max_tokens=300, temperature=0.1 to prevent token repetition loops, repeat_penalty=1.15
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.1,
                "extra_body": {
                    "options": {
                        "repeat_penalty": 1.15,
                    }
                },
            }
            if use_json_format:
                kwargs["response_format"] = {"type": "json_object"}
            response = active_client.chat.completions.create(**kwargs)
        elif active_backend == "gemini":
            # Gemini OpenAI-compatible endpoint (1500 tokens accommodates reasoning overhead)
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.0,
            }
            if use_json_format:
                kwargs["response_format"] = {"type": "json_object"}
            response = active_client.chat.completions.create(**kwargs)
        else:
            # Remote OpenCode Go: try with temperature=0.0 first
            try:
                response = active_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                err_str = str(e).lower()
                if "temperature" in err_str or "response_format" in err_str:
                    logger.info(f"{model} rejected temp/response_format, retrying plain")
                    response = active_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=1000,
                    )
                else:
                    raise

        choice = response.choices[0]
        msg = choice.message
        logger.info(f"Response finish_reason: {choice.finish_reason}")

        content = msg.content or ""
        if not content.strip():
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                logger.info(f"Content empty, using reasoning_content ({len(reasoning)} chars)")
                return reasoning

        return content.strip()

    def _reset_ollama_model(self, model: str | None = None) -> None:
        """Tell Ollama to unload the model from memory to unwedge its KV context."""
        target_model = model or self._ollama_model or self._model
        try:
            base_url = self._ollama_url.rstrip("/").removesuffix("/v1")
            req = urllib.request.Request(
                f"{base_url}/api/generate",
                data=json.dumps({"model": target_model, "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3.0):
                pass
            logger.info(f"Ollama model {target_model} unloaded from memory successfully.")
        except Exception as e:
            logger.warning(f"Could not reset Ollama model {target_model}: {e}")

    def _extract_via_glm_ocr(self, b64_image: str, client: OpenAI, ocr_model: str, json_model: str) -> ExtractedTicket:
        """Two-step local fallback: pure OCR image reading -> text-only JSON extraction."""
        logger.info(f"Trying hybrid OCR fallback: {ocr_model} -> {json_model}")
        ocr_resp = client.chat.completions.create(
            model=ocr_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                        {"type": "text", "text": "Text Recognition:"},
                    ],
                }
            ],
            max_tokens=800,
            temperature=0.0,
            extra_body={"options": {"repeat_penalty": 1.15}},
        )
        ocr_text = (ocr_resp.choices[0].message.content or "").strip()
        self._reset_ollama_model(ocr_model)
        if not ocr_text:
            raise RuntimeError(f"{ocr_model} returned empty OCR text")

        logger.info(f"{ocr_model} extracted {len(ocr_text)} chars; structuring via {json_model}...")
        json_resp = client.chat.completions.create(
            model=json_model,
            messages=[{"role": "user", "content": _build_text_ocr_prompt(ocr_text)}],
            max_tokens=350,
            temperature=0.1,
            extra_body={"options": {"repeat_penalty": 1.15}},
        )
        raw_json = (json_resp.choices[0].message.content or "").strip()
        logger.info(f"Hybrid OCR JSON response ({len(raw_json)} chars): {raw_json[:200]}")
        data = self._parse_response(raw_json)
        return ExtractedTicket(**data)

    def _extract_ollama(self, b64_image: str, client: OpenAI | None = None, model: str | None = None) -> ExtractedTicket:
        """Run Ollama extraction with self-healing resets and glm-ocr fallback."""
        active_client = client or self._client
        active_model = model or self._model
        logger.info(f"Trying ollama model: {active_model} ({len(b64_image)} chars b64)")

        raw = ""
        last_error: Exception | None = None
        attempts = [True, False, False]  # Step 1: JSON grammar; Step 2-3: plain after VRAM reset

        for idx, use_grammar in enumerate(attempts, start=1):
            try:
                raw = self._call_model(
                    active_model,
                    b64_image,
                    use_json_format=use_grammar,
                    client=active_client,
                    backend="ollama",
                )
                if raw.strip():
                    last_error = None
                    break
                logger.warning(f"Ollama call {idx} returned empty response.")
            except Exception as e:
                if self._is_connection_error(e):
                    raise OllamaUnavailableError(
                        f"Ollama local no está disponible en {self._ollama_url}"
                    ) from e
                last_error = e
                logger.warning(f"Ollama call {idx} failed ({type(e).__name__}: {e}).")

            if idx < len(attempts):
                logger.info("Resetting Ollama memory context before retry...")
                self._reset_ollama_model(active_model)
                time.sleep(0.5)

        if raw.strip():
            try:
                logger.info(f"Ollama response ({len(raw)} chars): {raw[:200]}")
                data = self._parse_response(raw)
                return ExtractedTicket(**data)
            except Exception as e:
                last_error = e
                logger.warning(f"Ollama JSON parsing failed: {e}")

        # All primary attempts failed (e.g. token repeat limit on handwritten grid): reset VRAM & try glm-ocr hybrid
        self._reset_ollama_model(active_model)
        installed = self._fetch_ollama_tags(self._ollama_url, timeout=1.5) or []
        ocr_model = next((m for m in self.OLLAMA_OCR_FALLBACK_MODELS if m in installed), None)
        if ocr_model and ocr_model != active_model:
            try:
                return self._extract_via_glm_ocr(b64_image, active_client, ocr_model, active_model)
            except Exception as hybrid_err:
                logger.warning(f"Hybrid OCR fallback failed: {hybrid_err}")
                self._reset_ollama_model(ocr_model)
                last_error = last_error or hybrid_err

        raise last_error or RuntimeError("Ollama extraction failed")

    @staticmethod
    def _classify_gemini_failure(exc: Exception) -> str:
        """Classify Gemini failure to decide permanent disable vs transient cooldown."""
        msg = f"{type(exc).__name__} {exc}".lower()
        if "notfounderror" in msg or "error code: 404" in msg or "404 " in msg:
            return "permanent"
        if any(
            token in msg
            for token in ("ratelimiterror", "429", "503", "unavailable", "timed out", "timeout", "overloaded")
        ):
            return "transient"
        return "other"

    def _call_gemini_pair(
        self,
        key_idx: int,
        client: OpenAI,
        model: str,
        b64_image: str,
    ) -> ExtractedTicket:
        """Invoke a single (API key, Gemini model) pair and parse result."""
        logger.info(f"Trying gemini model: {model} (key #{key_idx + 1}, {len(b64_image)} chars b64)")
        raw = self._call_model(model, b64_image, use_json_format=True, client=client, backend="gemini")
        if not raw.strip():
            raw = self._call_model(model, b64_image, use_json_format=False, client=client, backend="gemini")
        logger.info(f"gemini response ({model}, {len(raw)} chars): {raw[:200]}")
        data = self._parse_response(raw)
        return ExtractedTicket(**data)

    def _extract_gemini(self, b64_image: str) -> ExtractedTicket:
        """Run Gemini extraction across ordered healthy Flash/Flash-Lite fallbacks and API key pool."""
        models_to_try = [self._model] + [m for m in self.GEMINI_FALLBACK_MODELS if m != self._model]
        clients = self._gemini_clients or [self._client]
        last_error: Exception | None = None
        now = time.monotonic()

        cooled_candidates: list[tuple[float, int, str]] = []
        tried_in_call = 0

        # Order: try all healthy models across primary key first, then secondary keys
        for key_idx, client in enumerate(clients):
            for model in models_to_try:
                pair = (key_idx, model)
                if pair in self._disabled_gemini_pairs:
                    continue
                cooldown_until = self._gemini_cooldowns.get(pair, 0.0)
                if now < cooldown_until:
                    cooled_candidates.append((cooldown_until, key_idx, model))
                    continue

                tried_in_call += 1
                try:
                    return self._call_gemini_pair(key_idx, client, model, b64_image)
                except Exception as e:
                    last_error = e
                    kind = self._classify_gemini_failure(e)
                    if kind == "permanent":
                        self._disabled_gemini_pairs.add(pair)
                        logger.warning(f"Gemini {model} (key #{key_idx + 1}) returned 404; disabled for session.")
                    elif kind == "transient":
                        self._gemini_cooldowns[pair] = time.monotonic() + self.GEMINI_COOLDOWN_SECONDS
                        logger.warning(
                            f"Gemini {model} (key #{key_idx + 1}) transient failure ({type(e).__name__}); "
                            f"cooldown {int(self.GEMINI_COOLDOWN_SECONDS)}s."
                        )
                    else:
                        # Short cooldown to skip misbehaving model for the rest of a burst
                        self._gemini_cooldowns[pair] = time.monotonic() + 60.0
                        logger.warning(f"Gemini {model} (key #{key_idx + 1}) failed: {type(e).__name__}: {str(e)[:180]}")

        # If every active pair was skipped due to prior cooldowns, probe the earliest-expiring pair once
        if tried_in_call == 0 and cooled_candidates:
            cooled_candidates.sort(key=lambda item: item[0])
            for _, key_idx, model in cooled_candidates[:2]:
                self._gemini_cooldowns.pop((key_idx, model), None)
                try:
                    return self._call_gemini_pair(key_idx, clients[key_idx], model, b64_image)
                except Exception as e:
                    last_error = e
                    self._gemini_cooldowns[(key_idx, model)] = time.monotonic() + 60.0
                    logger.warning(f"Gemini recovery probe {model} failed: {type(e).__name__}: {str(e)[:150]}")

        if self._is_local and self._ollama_client is not None and self.is_ollama_available(self._ollama_url):
            logger.warning("All Gemini models failed; falling back to local Ollama...")
            return self._extract_ollama(b64_image, client=self._ollama_client, model=self._ollama_model)

        raise last_error or RuntimeError("All Gemini models failed")

    def _extract_sync(self, image_bytes: bytes) -> ExtractedTicket:
        """Synchronous extraction with fallback models."""
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        if self._backend == "ollama":
            return self._extract_ollama(b64_image)

        if self._backend == "gemini":
            return self._extract_gemini(b64_image)

        # Remote OpenCode Go: try primary then fallback models
        models_to_try = [self._model] + [m for m in self.REMOTE_FALLBACK_MODELS if m != self._model]
        last_error = None
        for model in models_to_try:
            try:
                logger.info(f"Trying {model} ({len(b64_image)} chars b64)")
                raw = self._call_model(model, b64_image)
                logger.info(f"AI response ({model}, {len(raw)} chars): {raw[:200]}")
                data = self._parse_response(raw)
                return ExtractedTicket(**data)
            except Exception as e:
                last_error = e
                logger.warning(f"{model} failed: {type(e).__name__}: {str(e)[:150]}")
                continue

        raise last_error or RuntimeError("All models failed")

    async def extract_from_image(self, image_bytes: bytes) -> ExtractedTicket:
        """Extract ticket data from an image with sequential concurrency control."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            return await asyncio.to_thread(self._extract_sync, image_bytes)

    def _parse_response(self, raw: str) -> dict:
        """Parse JSON response from AI, handling common markdown and formatting quirks."""
        # Try direct JSON parse
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return self._sanitize_data(data)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict):
                    return self._sanitize_data(data)
            except json.JSONDecodeError:
                pass

        # Try finding outermost JSON object
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    return self._sanitize_data(data)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Cannot parse AI response as JSON: {raw[:200]}")

    def _sanitize_data(self, data: dict) -> dict:
        """Apply strict sanitizers to each extracted field."""
        total = clean_total(data.get("total"))
        raw_es_ticket = data.get("es_ticket")
        tipo_doc = str(data.get("tipo_documento", "")).strip().lower()

        if tipo_doc == "no_ticket":
            es_ticket = False
        elif raw_es_ticket is None:
            es_ticket = total > 0.0
        elif isinstance(raw_es_ticket, bool):
            es_ticket = raw_es_ticket
        elif isinstance(raw_es_ticket, (int, float)):
            es_ticket = bool(raw_es_ticket)
        else:
            es_ticket = str(raw_es_ticket).strip().lower() in ("true", "1", "yes", "si", "sí")

        # A ticket with 0 or negative total cannot be considered a valid expense
        if total <= 0.0:
            es_ticket = False

        return {
            "es_ticket": es_ticket,
            "empresa": clean_empresa(data.get("empresa")) if es_ticket else "",
            "nif": clean_nif(data.get("nif")) if es_ticket else "",
            "descripcion": clean_descripcion(data.get("descripcion")) if es_ticket else "",
            "total": total if es_ticket else 0.0,
            "fecha": clean_fecha(data.get("fecha")) if es_ticket else "",
        }
