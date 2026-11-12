"""Tests for OCR AI extractor."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from ocr.extractor import (
    ExtractedTicket,
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
