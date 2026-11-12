"""Telegram keyboards for bot UI navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

if TYPE_CHECKING:
    from db.models import User

# Text constants for button matching
BTN_RESUMEN = "📊 Mi Cuenta / Saldo"
BTN_RESUMEN_LEGACY = "📊 Mi Resumen"
BTN_PANEL_JEFA = "👑 Panel Jefa (Saldos)"
BTN_ASIGNAR_DINERO = "💵 Asignar Dinero"
BTN_DEVOLVER = "💸 Devolver Sobrante"
BTN_BOLSILLO = "👛 Puse de mi Bolsillo"
BTN_SINCRONIZAR = "🔄 Sincronizar Excel"
BTN_AYUDA = "❓ Ayuda"
BTN_CANCELAR = "❌ Cancelar"
BTN_SIN_NOTA = "📝 Sin Nota"


def get_main_keyboard(is_admin: bool = False) -> ReplyKeyboardMarkup:
    """Return persistent bottom keypad with main actions depending on role."""
    if is_admin:
        keyboard = [
            [
                KeyboardButton(text=BTN_PANEL_JEFA),
                KeyboardButton(text=BTN_ASIGNAR_DINERO),
            ],
            [
                KeyboardButton(text=BTN_RESUMEN),
                KeyboardButton(text=BTN_DEVOLVER),
            ],
            [
                KeyboardButton(text=BTN_SINCRONIZAR),
                KeyboardButton(text=BTN_AYUDA),
            ],
        ]
    else:
        keyboard = [
            [
                KeyboardButton(text=BTN_RESUMEN),
                KeyboardButton(text=BTN_DEVOLVER),
            ],
            [
                KeyboardButton(text=BTN_BOLSILLO),
                KeyboardButton(text=BTN_SINCRONIZAR),
            ],
            [
                KeyboardButton(text=BTN_AYUDA),
            ],
        ]

    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="📸 Mandá una foto o elegí una opción...",
    )


def get_cancel_keyboard() -> ReplyKeyboardMarkup:
    """Single-button keyboard to cancel any active FSM dialog."""
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BTN_CANCELAR)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def get_note_keyboard() -> ReplyKeyboardMarkup:
    """Keyboard offering quick 'Sin Nota' button or cancel."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_SIN_NOTA)],
            [KeyboardButton(text=BTN_CANCELAR)],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def get_sync_inline_keyboard() -> InlineKeyboardMarkup:
    """Return inline button attached underneath messages when Excel is locked."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Reintentar Sincronización Ahora",
                    callback_data="do_sync",
                )
            ]
        ]
    )


def get_workers_inline_keyboard(workers: list[User]) -> InlineKeyboardMarkup:
    """Inline keyboard for admin to pick a worker to assign money to."""
    inline_keyboard: list[list[InlineKeyboardButton]] = []

    # 2 workers per row
    row: list[InlineKeyboardButton] = []
    for w in workers:
        role_tag = " (Admin)" if w.role == "admin" else ""
        btn_text = f"👤 {w.name}{role_tag}"
        row.append(InlineKeyboardButton(text=btn_text, callback_data=f"asig_worker:{w.id}"))
        if len(row) == 2:
            inline_keyboard.append(row)
            row = []
    if row:
        inline_keyboard.append(row)

    # Option to create a dummy worker for instant local testing without needing a 2nd phone
    inline_keyboard.append([
        InlineKeyboardButton(text="➕ Crear Trabajador de Prueba", callback_data="asig_create_dummy"),
    ])
    inline_keyboard.append([
        InlineKeyboardButton(text="❌ Cancelar", callback_data="cancel_action"),
    ])

    return InlineKeyboardMarkup(inline_keyboard=inline_keyboard)
