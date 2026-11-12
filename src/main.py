"""Main entry point for the Telegram ticket bot."""

from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand, BotCommandScopeDefault

from config import load_settings
from db.repository import Repository
from ocr.extractor import VisionExtractor
from bot.handlers import create_router
from excel.writer import ensure_workbook, sync_pending_tickets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def setup_bot_commands(bot: Bot) -> None:
    """Register bot commands in Telegram's native menu button."""
    commands = [
        BotCommand(command="resumen", description="📊 Ver mi cuenta y saldo neto"),
        BotCommand(command="panel", description="👑 Panel Jefa: saldos de trabajadores"),
        BotCommand(command="asignar", description="💵 Asignar dinero a un trabajador"),
        BotCommand(command="devolver", description="💸 Devolver sobrante a la jefa"),
        BotCommand(command="sincronizar", description="🔄 Forzar sincronización con Excel"),
        BotCommand(command="ayuda", description="❓ Ayuda e instrucciones de uso"),
        BotCommand(command="cancelar", description="❌ Cancelar acción en curso"),
    ]
    try:
        await bot.set_my_commands(commands, scope=BotCommandScopeDefault())
        logger.info("Telegram native command menu configured successfully.")
    except Exception as e:
        logger.warning(f"Could not set native Telegram commands: {e}")


async def main() -> None:
    """Start the bot."""
    # Load config from environment
    settings = load_settings()

    # Ensure data directories exist
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize components
    repo = Repository(settings.db_path)
    extractor = VisionExtractor(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        is_local=settings.is_local_ia,
        ollama_model=settings.ollama_model,
        ollama_url=settings.ollama_url,
        use_gemini=settings.use_gemini,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
    )

    if settings.use_gemini:
        mode = f"CLOUD (Google Gemini: {settings.gemini_model})"
    elif settings.is_local_ia:
        mode = f"LOCAL (Ollama: {settings.ollama_model})"
    else:
        mode = f"REMOTE (OpenCode Go: {settings.openai_model})"
    logger.info(f"AI mode: {mode}")

    # Ensure Excel file exists and sync any backlogged tickets
    ensure_workbook(settings.excel_path)
    synced, ok = sync_pending_tickets(settings.excel_path, repo)
    if synced > 0:
        logger.info(f"Startup: Synchronized {synced} pending tickets to Excel.")
    elif not ok:
        logger.warning("Startup: Excel file is locked; pending tickets will sync upon closing Excel.")

    # Create bot and dispatcher
    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    # Create and register router
    router = create_router(
        repo=repo,
        extractor=extractor,
        bot=bot,
        images_dir=settings.images_dir,
        excel_path=settings.excel_path,
        admin_telegram_ids=settings.admin_telegram_ids,
    )
    dp.include_router(router)

    logger.info("Bot started. Waiting for messages...")

    # Configure native Telegram menu button
    await setup_bot_commands(bot)

    # Start polling
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
