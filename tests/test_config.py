"""Tests for config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings, load_settings


class TestSettings:
    """Tests for the Settings dataclass."""

    def test_settings_creation_with_defaults(self):
        settings = Settings(
            telegram_bot_token="token123",
            openai_api_key="key456",
        )
        assert settings.telegram_bot_token == "token123"
        assert settings.openai_api_key == "key456"
        assert settings.openai_base_url == "https://api.openai.com/v1"
        assert settings.openai_model == "gpt-4o-mini"
        assert settings.is_local_ia is False
        assert settings.ollama_model == "glm-ocr"
        assert settings.db_path == Path("data/db.sqlite3")
        assert settings.excel_path == Path("data/tickets.xlsx")
        assert settings.images_dir == Path("data/images")

    def test_settings_local_ia(self):
        settings = Settings(
            telegram_bot_token="token",
            is_local_ia=True,
            ollama_model="glm-ocr",
        )
        assert settings.is_local_ia is True
        assert settings.ollama_model == "glm-ocr"
        assert settings.openai_api_key == ""  # Not needed for local

    def test_settings_paths_are_path_objects(self):
        settings = Settings(
            telegram_bot_token="t",
            db_path="data/test.db",
        )
        assert isinstance(settings.db_path, Path)
        assert isinstance(settings.excel_path, Path)
        assert isinstance(settings.images_dir, Path)

    def test_settings_is_frozen(self):
        settings = Settings(telegram_bot_token="t")
        with pytest.raises(AttributeError):
            settings.telegram_bot_token = "new"


class TestLoadSettings:
    """Tests for the load_settings function."""

    def test_load_settings_missing_telegram_token(self, tmp_path, monkeypatch):
        empty_env = tmp_path / ".env"
        empty_env.write_text("OPENAI_API_KEY=key\n")
        with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
            load_settings(env_path=empty_env)

    def test_load_settings_missing_openai_key_without_local(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("TELEGRAM_BOT_TOKEN=token\n")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("IS_LOCAL_IA", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            load_settings(env_path=env_file)

    def test_load_settings_local_ia_no_key_needed(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=token\n"
            "IS_LOCAL_IA=true\n"
            "OLLAMA_MODEL=glm-ocr\n"
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = load_settings(env_path=env_file)
        assert settings.is_local_ia is True
        assert settings.ollama_model == "glm-ocr"
        assert settings.openai_api_key == ""

    def test_load_settings_local_ia_case_insensitive(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=token\n"
            "IS_LOCAL_IA=True\n"
        )
        settings = load_settings(env_path=env_file)
        assert settings.is_local_ia is True

    def test_load_settings_remote_with_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=token\n"
            "OPENAI_API_KEY=key\n"
            "IS_LOCAL_IA=false\n"
        )
        settings = load_settings(env_path=env_file)
        assert settings.is_local_ia is False
        assert settings.openai_api_key == "key"

    def test_load_settings_defaults(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=token\n"
            "OPENAI_API_KEY=key\n"
        )
        settings = load_settings(env_path=env_file)
        assert settings.is_local_ia is False
        assert settings.ollama_model == "glm-ocr"

    def test_load_settings_custom_base_url(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=token\n"
            "OPENAI_API_KEY=key\n"
            "OPENAI_BASE_URL=https://opencode.ai/zen/go/v1\n"
        )
        settings = load_settings(env_path=env_file)
        assert settings.openai_base_url == "https://opencode.ai/zen/go/v1"

    def test_load_settings_base_url_strips_chat_completions(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=token\n"
            "OPENAI_API_KEY=key\n"
            "OPENAI_BASE_URL=https://opencode.ai/zen/go/v1/chat/completions\n"
        )
        settings = load_settings(env_path=env_file)
        assert settings.openai_base_url == "https://opencode.ai/zen/go/v1"

    def test_load_settings_gemini(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=token\n"
            "USE_GEMINI=true\n"
            "GEMINI_API_KEY=gemini-secret-123\n"
            "GEMINI_MODEL=gemini-2.0-flash\n"
        )
        settings = load_settings(env_path=env_file)
        assert settings.use_gemini is True
        assert settings.gemini_api_key == "gemini-secret-123"
        assert settings.gemini_model == "gemini-2.0-flash"

    def test_load_settings_gemini_missing_key_raises(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=token\n"
            "USE_GEMINI=true\n"
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            load_settings(env_path=env_file)

    def test_load_settings_admin_ids(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=token\n"
            "OPENAI_API_KEY=key\n"
            "ADMIN_TELEGRAM_IDS=12345, 67890\n"
        )
        settings = load_settings(env_path=env_file)
        assert settings.admin_telegram_ids == (12345, 67890)

    def test_load_settings_gemini_multi_keys(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "TELEGRAM_BOT_TOKEN=token\n"
            "USE_GEMINI=true\n"
            "GEMINI_API_KEY=key1\n"
            "GEMINI_API_KEYS=key1, key2, key3\n"
        )
        settings = load_settings(env_path=env_file)
        assert settings.gemini_api_key == "key1"
        assert settings.gemini_api_keys == ("key1", "key2", "key3")
