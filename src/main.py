"""Main entry point for the Telegram ticket bot."""

from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand, BotCommandScopeChat, BotCommandScopeDefault

from config import load_settings
from db.repository import Repository
from ocr.extractor import VisionExtractor
from bot.handlers import create_router
from excel.writer import ensure_workbook, rebuild_excel_from_db, sync_pending_tickets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def setup_bot_commands(
    bot: Bot,
    repo: Repository | None = None,
    admin_telegram_ids: tuple[int, ...] = (),
) -> None:
    """Register role-tailored commands in Telegram's native menu button."""
    worker_commands = [
        BotCommand(command="resumen", description="📊 Ver mi cuenta y saldo neto"),
        BotCommand(command="devolver", description="💸 Devolver sobrante a la empresa"),
        BotCommand(command="ayuda", description="❓ Ayuda e instrucciones de uso"),
        BotCommand(command="cancelar", description="❌ Cancelar acción en curso"),
    ]
    admin_commands = [
        BotCommand(command="panel", description="👑 Panel Admin: saldos de trabajadores"),
        BotCommand(command="asignar", description="💵 Asignar dinero a un trabajador"),
        BotCommand(command="resumen", description="📊 Ver mi cuenta y saldo neto"),
        BotCommand(command="devolver", description="💸 Devolver sobrante a la empresa"),
        BotCommand(command="sincronizar", description="🔄 Forzar sincronización con Excel"),
        BotCommand(command="excel", description="📥 Descargar archivo Excel actual"),
        BotCommand(command="ayuda", description="❓ Ayuda e instrucciones de uso"),
        BotCommand(command="cancelar", description="❌ Cancelar acción en curso"),
    ]
    admin_set = {tid for tid in admin_telegram_ids if tid > 0}
    try:
        await bot.set_my_commands(worker_commands, scope=BotCommandScopeDefault())
        for admin_tid in admin_set:
            try:
                await bot.set_my_commands(
                    admin_commands,
                    scope=BotCommandScopeChat(chat_id=admin_tid),
                )
            except Exception as e:
                logger.warning(f"Could not set admin Telegram commands for {admin_tid}: {e}")
        if repo is not None:
            for u in repo.get_all_users():
                if u.telegram_id > 0 and u.telegram_id not in admin_set:
                    try:
                        await bot.delete_my_commands(scope=BotCommandScopeChat(chat_id=u.telegram_id))
                    except Exception:
                        pass
        logger.info("Telegram native command menu configured successfully.")
    except Exception as e:
        logger.warning(f"Could not set native Telegram commands: {e}")


async def main() -> None:
    """Start the bot."""
    # Load config from environment
    settings = load_settings()

    # Ensure data and export directories exist
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.excel_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize components
    repo = Repository(settings.db_path)

    # Enforce .env ADMIN_TELEGRAM_IDS as the single source of truth on startup
    promoted, demoted = repo.sync_admin_roles(settings.admin_telegram_ids)
    if promoted > 0 or demoted > 0:
        logger.info(
            f"Startup: Roles synced from ADMIN_TELEGRAM_IDS "
            f"(promoted: {promoted}, demoted: {demoted})."
        )

    extractor = VisionExtractor(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        is_local=settings.is_local_ia,
        ollama_model=settings.ollama_model,
        ollama_url=settings.ollama_url,
        use_gemini=settings.use_gemini,
        gemini_api_key=settings.gemini_api_key,
        gemini_api_keys=settings.gemini_api_keys,
        gemini_model=settings.gemini_model,
    )

    if settings.use_gemini:
        mode = f"CLOUD (Google Gemini: {settings.gemini_model}, fallback automático activo)"
    elif settings.is_local_ia:
        mode = f"LOCAL (Ollama: {settings.ollama_model})"
    else:
        mode = f"REMOTE (OpenCode Go: {settings.openai_model})"
    logger.info(f"AI mode: {mode}")

    if settings.is_local_ia and not VisionExtractor.is_ollama_available(settings.ollama_url):
        if settings.use_gemini:
            logger.info("Ollama local no está iniciado (solo se usará Gemini cloud).")
        else:
            logger.warning(
                f"⚠️ ATENCIÓN: Ollama local no responde en {settings.ollama_url}. "
                "Asegurate de abrir Ollama antes de procesar tickets."
            )

    # Ensure Excel file exists and sync any backlogged tickets or updated roles
    ensure_workbook(settings.excel_path)
    if promoted > 0 or demoted > 0:
        _, rebuilt_ok = rebuild_excel_from_db(settings.excel_path, repo)
        if rebuilt_ok:
            logger.info("Startup: Rebuilt Excel to reflect updated admin/worker roles.")
        else:
            logger.warning("Startup: Excel file is locked; role updates will sync when Excel is closed.")
    else:
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

    # Configure native Telegram menu button per role
    await setup_bot_commands(bot, repo=repo, admin_telegram_ids=settings.admin_telegram_ids)

    # Start polling
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
