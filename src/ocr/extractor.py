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
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


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

class VisionExtractor:
    """Extracts ticket data using Vision API (Gemini, Ollama local, or OpenCode Go)."""

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
        gemini_model: str = "gemini-2.0-flash",
    ) -> None:
        self._use_gemini = use_gemini
        self._is_local = is_local
        self._ollama_model = ollama_model
        self._ollama_url = ollama_url
        self._lock: asyncio.Lock | None = None

        if use_gemini:
            self._backend = "gemini"
            self._model = gemini_model
            self._client = OpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=30.0,
            )
            logger.info(f"Using Google Gemini: {gemini_model}")
        elif is_local:
            self._backend = "ollama"
            self._model = ollama_model
            self._client = OpenAI(
                api_key="ollama",
                base_url=ollama_url,
                timeout=120.0,
            )
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

    def _call_model(self, model: str, b64_image: str, use_json_format: bool = True) -> str:
        """Call a single model with the image and appropriate parameters."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ]

        if self._backend == "ollama":
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
            response = self._client.chat.completions.create(**kwargs)
        elif self._backend == "gemini":
            # Gemini OpenAI-compatible endpoint
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.0,
            }
            if use_json_format:
                kwargs["response_format"] = {"type": "json_object"}
            response = self._client.chat.completions.create(**kwargs)
        else:
            # Remote OpenCode Go: try with temperature=0.0 first
            try:
                response = self._client.chat.completions.create(
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
                    response = self._client.chat.completions.create(
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

    def _reset_ollama_model(self) -> None:
        """Tell Ollama to unload the model from memory to unwedge its KV context."""
        try:
            import urllib.request
            base_url = self._ollama_url.rstrip("/").removesuffix("/v1")
            req = urllib.request.Request(
                f"{base_url}/api/generate",
                data=json.dumps({"model": self._model, "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5.0):
                pass
            logger.info(f"Ollama model {self._model} unloaded from memory successfully.")
        except Exception as e:
            logger.warning(f"Could not reset Ollama model: {e}")

    def _extract_sync(self, image_bytes: bytes) -> ExtractedTicket:
        """Synchronous extraction with fallback models."""
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        if self._backend == "ollama":
            logger.info(f"Trying ollama model: {self._model} ({len(b64_image)} chars b64)")
            raw = ""
            # Attempt 1: Call with grammar (json_object)
            try:
                raw = self._call_model(self._model, b64_image, use_json_format=True)
            except Exception as e:
                logger.warning(f"Ollama call 1 with grammar failed ({type(e).__name__}: {e}). Resetting memory context...")
                self._reset_ollama_model()
                time.sleep(1.0)

            # Attempt 2: If empty or failed, retry without grammar constraint
            if not raw.strip():
                try:
                    logger.warning("Ollama attempting without grammar constraint...")
                    raw = self._call_model(self._model, b64_image, use_json_format=False)
                except Exception as e:
                    logger.warning(f"Ollama call 2 without grammar failed ({type(e).__name__}: {e}). Resetting memory context...")
                    self._reset_ollama_model()
                    time.sleep(1.0)

            # Attempt 3: If still empty after reset, one final attempt without grammar
            if not raw.strip():
                try:
                    logger.warning("Ollama final retry after fresh memory reset...")
                    raw = self._call_model(self._model, b64_image, use_json_format=False)
                except Exception as e:
                    logger.error(f"Ollama final retry failed: {e}. Resetting memory to prevent poisoning subsequent requests.")
                    self._reset_ollama_model()
                    raise

            logger.info(f"Ollama response ({len(raw)} chars): {raw[:200]}")
            data = self._parse_response(raw)
            return ExtractedTicket(**data)

        if self._backend == "gemini":
            logger.info(f"Trying gemini model: {self._model} ({len(b64_image)} chars b64)")
            raw = self._call_model(self._model, b64_image, use_json_format=True)
            logger.info(f"gemini response ({len(raw)} chars): {raw[:200]}")
            data = self._parse_response(raw)
            return ExtractedTicket(**data)

        # Remote OpenCode Go: try primary then fallback models
        models_to_try = [self._model]
        for m in self.REMOTE_FALLBACK_MODELS:
            if m != self._model:
                models_to_try.append(m)

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
