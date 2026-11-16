"""Telegram native command menu configuration per user role."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiogram import Bot
from aiogram.types import BotCommand, BotCommandScopeChat, BotCommandScopeDefault

if TYPE_CHECKING:
    from db.repository import Repository

logger = logging.getLogger(__name__)

WORKER_COMMANDS = [
    BotCommand(command="resumen", description="📊 Ver mi cuenta y saldo neto"),
    BotCommand(command="devolver", description="💸 Devolver sobrante a la empresa"),
    BotCommand(command="ayuda", description="❓ Ayuda e instrucciones de uso"),
    BotCommand(command="cancelar", description="❌ Cancelar acción en curso"),
]

ADMIN_COMMANDS = [
    BotCommand(command="panel", description="👑 Panel Admin: saldos de trabajadores"),
    BotCommand(command="asignar", description="💵 Asignar dinero a un trabajador"),
    BotCommand(command="resumen", description="📊 Ver mi cuenta y saldo neto"),
    BotCommand(command="devolver", description="💸 Devolver sobrante a la empresa"),
    BotCommand(command="sincronizar", description="🔄 Forzar sincronización con Excel"),
    BotCommand(command="excel", description="📥 Descargar archivo Excel actual"),
    BotCommand(command="ayuda", description="❓ Ayuda e instrucciones de uso"),
    BotCommand(command="cancelar", description="❌ Cancelar acción en curso"),
]


async def sync_user_chat_commands(bot: Bot, telegram_id: int, is_admin: bool) -> None:
    """Update or clear chat-specific Telegram slash menu commands for one real user."""
    if telegram_id <= 0:
        return
    try:
        if is_admin:
            await bot.set_my_commands(ADMIN_COMMANDS, scope=BotCommandScopeChat(chat_id=telegram_id))
        else:
            await bot.delete_my_commands(scope=BotCommandScopeChat(chat_id=telegram_id))
    except Exception as e:
        logger.debug(f"Could not update Telegram commands for {telegram_id}: {e}")


async def setup_bot_commands(
    bot: Bot,
    repo: Repository | None = None,
    admin_telegram_ids: tuple[int, ...] = (),
) -> None:
    """Register role-tailored commands in Telegram's native menu button."""
    admin_set = {tid for tid in admin_telegram_ids if tid > 0}
    try:
        await bot.set_my_commands(WORKER_COMMANDS, scope=BotCommandScopeDefault())
        synced_chats: set[int] = set()
        for admin_tid in admin_set:
            await sync_user_chat_commands(bot, admin_tid, is_admin=True)
            synced_chats.add(admin_tid)

        if repo is not None:
            for u in repo.get_all_users():
                if u.telegram_id <= 0 or u.telegram_id in synced_chats:
                    continue
                is_bot_admin = u.role == "admin" and u.admin_source == "bot"
                await sync_user_chat_commands(bot, u.telegram_id, is_admin=is_bot_admin)

        logger.info("Telegram native command menu configured successfully.")
    except Exception as e:
        logger.warning(f"Could not set native Telegram commands: {e}")
