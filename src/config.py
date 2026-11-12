"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    telegram_bot_token: str
    admin_telegram_ids: tuple[int, ...] = ()
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    is_local_ia: bool = False
    ollama_model: str = "glm-ocr"
    ollama_url: str = "http://localhost:11434/v1"
    use_gemini: bool = False
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    db_path: Path = field(default_factory=lambda: Path("data/db.sqlite3"))
    excel_path: Path = field(default_factory=lambda: Path("data/tickets.xlsx"))
    images_dir: Path = field(default_factory=lambda: Path("data/images"))

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        object.__setattr__(self, "db_path", Path(self.db_path))
        object.__setattr__(self, "excel_path", Path(self.excel_path))
        object.__setattr__(self, "images_dir", Path(self.images_dir))


def load_settings(env_path: str | Path | None = None) -> Settings:
    """Load settings from environment variables.

    Args:
        env_path: Optional path to .env file. If None, uses default dotenv lookup.

    Returns:
        Settings instance with values from environment.

    Raises:
        ValueError: If required environment variables are missing.
    """
    if env_path is not None:
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=False)

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    is_local = os.getenv("IS_LOCAL_IA", "false").lower() in ("true", "1", "yes")
    use_gemini = os.getenv("USE_GEMINI", os.getenv("IS_GEMINI", "false")).lower() in ("true", "1", "yes")
    gemini_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

    if not telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    if use_gemini and not gemini_key:
        raise ValueError("GEMINI_API_KEY is required when USE_GEMINI=true")
    if not is_local and not use_gemini and not openai_key:
        raise ValueError("OPENAI_API_KEY is required (set IS_LOCAL_IA=true or USE_GEMINI=true)")

    raw_admin_ids = os.getenv("ADMIN_TELEGRAM_IDS", "")
    admin_ids: list[int] = []
    if raw_admin_ids:
        for piece in raw_admin_ids.split(","):
            piece = piece.strip()
            if piece.isdigit() or (piece.startswith("-") and piece[1:].isdigit()):
                admin_ids.append(int(piece))

    raw_db_path = Path(os.getenv("DB_PATH", "data/db.sqlite3"))
    raw_excel_path = Path(os.getenv("EXCEL_PATH", "data/tickets.xlsx"))
    raw_images_dir = Path(os.getenv("IMAGES_DIR", "data/images"))

    db_path = raw_db_path if raw_db_path.is_absolute() else (PROJECT_ROOT / raw_db_path)
    excel_path = raw_excel_path if raw_excel_path.is_absolute() else (PROJECT_ROOT / raw_excel_path)
    images_dir = raw_images_dir if raw_images_dir.is_absolute() else (PROJECT_ROOT / raw_images_dir)

    return Settings(
        telegram_bot_token=telegram_token,
        admin_telegram_ids=tuple(admin_ids),
        openai_api_key=openai_key,
        openai_base_url=_normalize_base_url(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        is_local_ia=is_local,
        ollama_model=os.getenv("OLLAMA_MODEL", "glm-ocr"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        use_gemini=use_gemini,
        gemini_api_key=gemini_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        db_path=db_path,
        excel_path=excel_path,
        images_dir=images_dir,
    )


def _normalize_base_url(url: str) -> str:
    """Normalize OpenAI base URL."""
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url
