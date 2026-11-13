"""Tests for OCR AI extractor."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from ocr.extractor import (
    ExtractedTicket,
    OllamaUnavailableError,
    VisionExtractor,
    clean_descripcion,
    clean_empresa,
    clean_fecha,
    clean_nif,
    clean_total,
    is_suspicious_total,
    is_valid_ticket,
)


class TestExtractedTicket:
    def test_creation(self):
        ticket = ExtractedTicket(
            empresa="Repsol",
            nif="B12345678",
            descripcion="Gasolina 95",
            total=36.89,
            fecha="15/08/2026",
        )
        assert ticket.empresa == "Repsol"
        assert ticket.nif == "B12345678"
        assert ticket.total == 36.89

    def test_creation_without_nif(self):
        ticket = ExtractedTicket(
            empresa="Bar",
            nif="",
            descripcion="Café",
            total=1.5,
            fecha="18/08/2026",
        )
        assert ticket.nif == ""


class TestSanitizers:
    """Tests for pure sanitization helpers."""

    def test_clean_total_numbers_and_strings(self):
        assert clean_total(24.5) == 24.5
        assert clean_total(24) == 24.0
        assert clean_total("7,30") == 7.30
        assert clean_total("35,50 €") == 35.50
        assert clean_total(" 24.00 EUR ") == 24.00
        assert clean_total("1.234,56") == 1234.56
        assert clean_total("1,234.56") == 1234.56
        assert clean_total("24'00") == 24.0  # Spanish handwritten apostrophe
        assert clean_total("") == 0.0
        assert clean_total(None) == 0.0
        assert clean_total("invalid") == 0.0
        assert clean_total(-10.0) == 0.0

    def test_clean_fecha_formats_and_placeholders(self):
        today_str = datetime.now().strftime("%d/%m/%Y")
        assert clean_fecha("") == today_str
        assert clean_fecha("DD/MM/YYYY") == today_str
        assert clean_fecha("dd/mm/yyyy") == today_str
        assert clean_fecha("HOY") == today_str
        assert clean_fecha("FECHA") == today_str
        assert clean_fecha("21/05/2024") == "21/05/2024"
        assert clean_fecha("21-05-2024") == "21/05/2024"
        assert clean_fecha("2024-05-21") == "21/05/2024"
        assert clean_fecha("21/05/24") == "21/05/2024"
        # Spanish text month
        assert clean_fecha("21 de Mayo de 2024") == "21/05/2024"
        assert clean_fecha("21 mayo 2024") == "21/05/2024"

    def test_clean_nif(self):
        assert clean_nif("B-23.742.901") == "B23742901"
        assert clean_nif("b14472021") == "B14472021"
        assert clean_nif("26742617B") == "26742617B"
        assert clean_nif("X1234567A") == "X1234567A"
        # Pure digits (missing letter) are rejected
        assert clean_nif("183122449") == ""
        assert clean_nif("") == ""
        assert clean_nif(None) == ""
        assert clean_nif("ABC") == ""  # Too short

    def test_clean_empresa(self):
        assert clean_empresa(' "Mesón La Casilla" ') == "Mesón La Casilla"
        assert clean_empresa("") == "Desconocido"
        assert clean_empresa("DESCONOCIDO") == "Desconocido"

    def test_clean_descripcion(self):
        assert clean_descripcion("Café + tostada") == "Café + tostada"
        assert clean_descripcion("Línea 1\nLínea 2") == "Ticket"
        assert clean_descripcion("") == "Ticket"

    def test_is_suspicious_total(self):
        assert is_suspicious_total(24400.0) is True
        assert is_suspicious_total(0.0) is True
        assert is_suspicious_total(-5.0) is True
        assert is_suspicious_total(24.0) is False
        assert is_suspicious_total(999.0) is False
        assert is_suspicious_total(1000.0) is True

    def test_is_valid_ticket(self):
        valid = ExtractedTicket(
            empresa="Repsol", nif="B123", descripcion="Gasolina",
            total=35.50, fecha="01/01/2026", es_ticket=True,
        )
        assert is_valid_ticket(valid) is True

        # False if total is 0.0
        zero_total = ExtractedTicket(
            empresa="Desconocido", nif="", descripcion="Ticket",
            total=0.0, fecha="01/01/2026", es_ticket=True,
        )
        assert is_valid_ticket(zero_total) is False

        # False if es_ticket is False
        non_ticket = ExtractedTicket(
            empresa="Screenshot", nif="", descripcion="Chat",
            total=50.0, fecha="01/01/2026", es_ticket=False,
        )
        assert is_valid_ticket(non_ticket) is False


class TestVisionExtractorParsing:
    @pytest.fixture
    def extractor(self):
        return VisionExtractor(api_key="test-key", model="kimi-k2.6")

    def test_parse_clean_json(self, extractor):
        raw = json.dumps({
            "empresa": "Repsol",
            "nif": "B12345678",
            "descripcion": "Gasolina 95",
            "total": 36.89,
            "fecha": "15/08/2026",
        })
        result = extractor._parse_response(raw)
        assert result["empresa"] == "Repsol"
        assert result["nif"] == "B12345678"
        assert result["total"] == 36.89

    def test_parse_json_in_markdown_block(self, extractor):
        raw = '```json\n{"empresa": "Hotel", "nif": "", "descripcion": "Hab", "total": "80,00", "fecha": "16/08/2026"}\n```'
        result = extractor._parse_response(raw)
        assert result["empresa"] == "Hotel"
        assert result["total"] == 80.0

    def test_parse_fill_missing_and_sanitizes(self, extractor):
        raw = '{"empresa": "Test", "total": "7,30", "fecha": "DD/MM/YYYY"}'
        result = extractor._parse_response(raw)
        assert result["empresa"] == "Test"
        assert result["nif"] == ""
        assert result["descripcion"] == "Ticket"
        assert result["total"] == 7.30
        assert result["fecha"] != "DD/MM/YYYY"  # Normalized to today

    def test_parse_invalid_json_raises(self, extractor):
        with pytest.raises(ValueError, match="Cannot parse"):
            extractor._parse_response("not json")


class TestVisionExtractorAPI:
    def test_ollama_parameters(self):
        ext = VisionExtractor(is_local=True, ollama_model="qwen2.5vl:3b")
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content='{"empresa": "Bar", "nif": "B12345678", "descripcion": "Café", "total": 1.5, "fecha": "01/01/2026"}'),
                finish_reason="stop",
            )
        ]

        with patch.object(ext._client.chat.completions, "create", return_value=mock_response) as mock_create:
            result = ext._extract_sync(b"fake-bytes")

        assert result.total == 1.5
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 300
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["extra_body"]["options"]["repeat_penalty"] == 1.15
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_gemini_backend_init(self):
        ext = VisionExtractor(use_gemini=True, gemini_api_key="gemini-key", gemini_model="gemini-2.0-flash")
        assert ext._backend == "gemini"
        assert ext._model == "gemini-2.0-flash"
        assert "generativelanguage.googleapis.com" in str(ext._client.base_url)

    def test_ollama_recovers_and_resets_on_failure(self):
        ext = VisionExtractor(is_local=True, ollama_model="qwen2.5vl:3b")
        mock_success = MagicMock()
        mock_success.choices = [
            MagicMock(
                message=MagicMock(content='{"tipo_documento": "ticket", "es_ticket": true, "empresa": "Bar", "nif": "B123", "descripcion": "Café", "total": 24.0, "fecha": "01/01/2026"}'),
                finish_reason="stop",
            )
        ]

        with patch.object(ext, "_reset_ollama_model") as mock_reset, \
             patch("time.sleep"), \
             patch.object(ext._client.chat.completions, "create", side_effect=[Exception("prediction aborted"), mock_success]):
            result = ext._extract_sync(b"fake-bytes")

        assert result.total == 24.0
        assert result.empresa == "Bar"
        assert mock_reset.called

    def test_gemini_fallback_on_503(self):
        ext = VisionExtractor(use_gemini=True, gemini_api_key="gemini-key", gemini_model="gemini-3.8-flash")
        mock_success = MagicMock()
        mock_success.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"tipo_documento": "ticket", "es_ticket": true, "empresa": "Casa Pepe", "nif": "75103486E", "descripcion": "Desayunos", "total": 10.0, "fecha": "27/10/2020"}'
                ),
                finish_reason="stop",
            )
        ]

        with patch.object(
            ext._client.chat.completions,
            "create",
            side_effect=[Exception("503 Service Unavailable"), mock_success],
        ) as mock_create:
            result = ext._extract_sync(b"fake-bytes")

        assert result.empresa == "Casa Pepe"
        assert result.total == 10.0
        assert mock_create.call_count == 2
        assert mock_create.call_args_list[0].kwargs["model"] == "gemini-3.8-flash"
        assert mock_create.call_args_list[1].kwargs["model"] == "gemini-3.5-flash-lite"

    def test_gemini_falls_back_to_local_ollama_if_all_fail(self):
        ext = VisionExtractor(
            use_gemini=True,
            gemini_api_key="gemini-key",
            gemini_model="gemini-3.8-flash",
            is_local=True,
            ollama_model="qwen2.5vl:3b",
        )
        mock_ollama_ok = MagicMock()
        mock_ollama_ok.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"tipo_documento": "ticket", "es_ticket": true, "empresa": "Venta Bonanza", "nif": "", "descripcion": "Desayuno", "total": 7.0, "fecha": "23/09/2023"}'
                ),
                finish_reason="stop",
            )
        ]

        with patch.object(ext._client.chat.completions, "create", side_effect=Exception("503 Unavailable")), \
             patch.object(VisionExtractor, "is_ollama_available", return_value=True), \
             patch.object(ext._ollama_client.chat.completions, "create", return_value=mock_ollama_ok):
            result = ext._extract_sync(b"fake-bytes")

        assert result.empresa == "Venta Bonanza"
        assert result.total == 7.0

    def test_ollama_raises_unavailable_immediately_on_connection_error(self):
        ext = VisionExtractor(is_local=True, ollama_model="qwen2.5vl:3b")

        with patch.object(ext, "_reset_ollama_model") as mock_reset, \
             patch.object(
                 ext._client.chat.completions,
                 "create",
                 side_effect=Exception("APIConnectionError: [WinError 10061] Connection refused"),
             ) as mock_create:
            with pytest.raises(OllamaUnavailableError):
                ext._extract_sync(b"fake-bytes")

        assert mock_create.call_count == 1
        assert not mock_reset.called

    def test_ollama_hybrid_glm_ocr_fallback(self):
        ext = VisionExtractor(is_local=True, ollama_model="qwen2.5vl:3b")
        mock_ocr_text = MagicMock()
        mock_ocr_text.choices = [
            MagicMock(message=MagicMock(content="FACTURA 2 Desayunos 10,00 27/10/2020"), finish_reason="stop")
        ]
        mock_json = MagicMock()
        mock_json.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"tipo_documento": "ticket", "es_ticket": true, "empresa": "Bar Pepe", "nif": "", "descripcion": "2 Desayunos", "total": 10.0, "fecha": "27/10/2020"}'
                ),
                finish_reason="stop",
            )
        ]

        with patch.object(ext, "_reset_ollama_model") as mock_reset, \
             patch("time.sleep"), \
             patch.object(VisionExtractor, "_fetch_ollama_tags", return_value=["qwen2.5vl:3b", "glm-ocr:latest"]), \
             patch.object(
                 ext._client.chat.completions,
                 "create",
                 side_effect=[
                     Exception("token repeat limit reached"),
                     Exception("token repeat limit reached"),
                     Exception("token repeat limit reached"),
                     mock_ocr_text,
                     mock_json,
                 ],
             ):
            result = ext._extract_sync(b"fake-bytes")

        assert result.total == 10.0
        assert result.empresa == "Bar Pepe"
        assert mock_reset.called

    def test_gemini_circuit_breaker_skips_404_and_cooldown_models(self):
        ext = VisionExtractor(
            use_gemini=True,
            gemini_api_key="key1",
            gemini_model="gemini-3.5-flash-lite",
        )
        ok_resp = MagicMock()
        ok_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"tipo_documento": "ticket", "es_ticket": true, "empresa": "Mercadona", "nif": "", "descripcion": "Compra", "total": 12.5, "fecha": "13/11/2026"}'
                ),
                finish_reason="stop",
            )
        ]

        # Call 1: primary returns 404 (disabled for session), second returns 429 (cooldown), third succeeds
        with patch.object(
            ext._client.chat.completions,
            "create",
            side_effect=[
                Exception("NotFoundError: Error code: 404"),
                Exception("RateLimitError: Error code: 429"),
                ok_resp,
            ],
        ) as mock_call_1:
            r1 = ext._extract_sync(b"img1")

        assert r1.empresa == "Mercadona"
        assert mock_call_1.call_count == 3

        # Call 2: must jump directly to 3rd model without re-calling disabled/cooldown models
        with patch.object(ext._client.chat.completions, "create", return_value=ok_resp) as mock_call_2:
            r2 = ext._extract_sync(b"img2")

        assert r2.total == 12.5
        assert mock_call_2.call_count == 1
        assert mock_call_2.call_args.kwargs["model"] == VisionExtractor.GEMINI_FALLBACK_MODELS[2]

    def test_gemini_multi_key_pool_rotates_on_failure(self):
        ext = VisionExtractor(
            use_gemini=True,
            gemini_api_keys=("key-one", "key-two"),
            gemini_model="gemini-3.5-flash-lite",
        )
        assert len(ext._gemini_clients) == 2
        ok_resp = MagicMock()
        ok_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"tipo_documento": "ticket", "es_ticket": true, "empresa": "Repsol", "nif": "", "descripcion": "Fuel", "total": 45.0, "fecha": "13/11/2026"}'
                ),
                finish_reason="stop",
            )
        ]

        with patch.object(ext._gemini_clients[0].chat.completions, "create", side_effect=Exception("429 Quota Exceeded")) as c0, \
             patch.object(ext._gemini_clients[1].chat.completions, "create", return_value=ok_resp) as c1:
            result = ext._extract_sync(b"img")

        assert result.empresa == "Repsol"
        assert result.total == 45.0
        assert c0.called
        assert c1.call_count == 1
