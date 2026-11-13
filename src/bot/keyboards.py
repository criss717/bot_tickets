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
    from db.models import CashMovement, Ticket, User

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

ALL_MENU_BUTTONS = frozenset({
    BTN_RESUMEN,
    BTN_RESUMEN_LEGACY,
    BTN_PANEL_JEFA,
    BTN_ASIGNAR_DINERO,
    BTN_DEVOLVER,
    BTN_BOLSILLO,
    BTN_SINCRONIZAR,
    BTN_AYUDA,
    BTN_CANCELAR,
})


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


def get_workers_inline_keyboard(
    workers: list[User],
    callback_prefix: str = "asig_worker",
    include_create_dummy: bool = True,
) -> InlineKeyboardMarkup:
    """Inline keyboard for admin to pick a worker for money assignment or management."""
    inline_keyboard: list[list[InlineKeyboardButton]] = []

    row: list[InlineKeyboardButton] = []
    for w in workers:
        role_tag = " (Admin)" if w.role == "admin" else ""
        btn_text = f"👤 {w.name}{role_tag}"
        row.append(InlineKeyboardButton(text=btn_text, callback_data=f"{callback_prefix}:{w.id}"))
        if len(row) == 2:
            inline_keyboard.append(row)
            row = []
    if row:
        inline_keyboard.append(row)

    if include_create_dummy:
        inline_keyboard.append([
            InlineKeyboardButton(text="➕ Crear Trabajador de Prueba", callback_data="asig_create_dummy"),
        ])
    inline_keyboard.append([
        InlineKeyboardButton(text="❌ Cancelar", callback_data="cancel_action"),
    ])

    return InlineKeyboardMarkup(inline_keyboard=inline_keyboard)


def get_amount_inline_keyboard(
    prefix: str,
    amounts: tuple[int | float, ...] = (20, 50, 100, 150, 200),
    extra_top_button: tuple[str, str] | None = None,
) -> InlineKeyboardMarkup:
    """Inline quick-amount buttons to speed up entering common Euro amounts."""
    rows: list[list[InlineKeyboardButton]] = []
    if extra_top_button is not None:
        label, cb_data = extra_top_button
        rows.append([InlineKeyboardButton(text=label, callback_data=cb_data)])

    row: list[InlineKeyboardButton] = []
    for amt in amounts:
        val_str = f"{int(amt)}" if float(amt).is_integer() else f"{amt:.2f}"
        label = f"{val_str} €"
        row.append(InlineKeyboardButton(text=label, callback_data=f"{prefix}:{amt}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    rows.append([InlineKeyboardButton(text="❌ Cancelar", callback_data="cancel_action")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_boss_panel_inline_keyboard() -> InlineKeyboardMarkup:
    """Admin action toolbar attached beneath the Boss Control Panel."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📥 Descargar Excel", callback_data="excel_download"),
            ],
            [
                InlineKeyboardButton(text="↩️ Desasignar / Anular Mov.", callback_data="admin_unassign_menu"),
                InlineKeyboardButton(text="🧾 Gestionar Tickets", callback_data="admin_tickets_menu"),
            ],
            [
                InlineKeyboardButton(text="👥 Gestionar Trabajadores", callback_data="admin_workers_menu"),
            ],
        ]
    )


def get_unassign_options_keyboard(worker_id: int) -> InlineKeyboardMarkup:
    """Inline options after selecting a worker to unassign or void cash movements."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="➖ Desasignar importe (Restar saldo)", callback_data=f"unassign_custom:{worker_id}")],
            [InlineKeyboardButton(text="🗑️ Anular un movimiento concreto", callback_data=f"unassign_void_list:{worker_id}")],
            [InlineKeyboardButton(text="❌ Cancelar", callback_data="cancel_action")],
        ]
    )


def get_movements_delete_keyboard(movements: list[tuple[CashMovement, str, str]]) -> InlineKeyboardMarkup:
    """Inline keyboard listing recent cash movements to void one."""
    type_icons = {
        "ASIGNACION": "💵+",
        "DESASIGNACION": "↩️-",
        "DEVOLUCION": "💸",
        "APORTE_PROPIO": "👛",
    }
    rows: list[list[InlineKeyboardButton]] = []
    for mov, _worker_name, _creator_name in movements:
        icon = type_icons.get(mov.movement_type, "•")
        note_snip = f" ({mov.note[:14]})" if mov.note else ""
        label = f"🗑️ {mov.fecha} {icon} {mov.amount:.2f} €{note_snip}"
        rows.append([InlineKeyboardButton(text=label, callback_data=f"mov_del:{mov.id}")])
    rows.append([InlineKeyboardButton(text="❌ Cancelar", callback_data="cancel_action")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_ticket_actions_inline_keyboard(ticket_id: int) -> InlineKeyboardMarkup:
    """Inline quick correction/void buttons beneath a processed ticket."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✏️ Corregir Total", callback_data=f"tkt_edit:{ticket_id}"),
                InlineKeyboardButton(text="🗑️ Anular Ticket", callback_data=f"tkt_del:{ticket_id}"),
            ]
        ]
    )


def get_confirm_delete_ticket_keyboard(ticket_id: int) -> InlineKeyboardMarkup:
    """Confirmation dialog before deleting a ticket."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Sí, anular ticket", callback_data=f"tkt_del_confirm:{ticket_id}"),
                InlineKeyboardButton(text="❌ No, mantener", callback_data="cancel_action"),
            ]
        ]
    )


def get_manage_tickets_keyboard(tickets: list[tuple[Ticket, str]]) -> InlineKeyboardMarkup:
    """Inline list of recent tickets for admin inspection/correction."""
    rows: list[list[InlineKeyboardButton]] = []
    for t, user_name in tickets:
        short_emp = (t.empresa or "Ticket")[:14]
        label = f"🧾 #{t.id} {t.fecha} · {user_name[:10]} · {short_emp} · {t.total:.2f} €"
        rows.append([InlineKeyboardButton(text=label, callback_data=f"tkt_pick:{t.id}")])
    rows.append([InlineKeyboardButton(text="❌ Cerrar", callback_data="cancel_action")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_worker_actions_keyboard(worker: User, is_env_superadmin: bool = False) -> InlineKeyboardMarkup:
    """Inline management options for a selected worker."""
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text="✏️ Renombrar trabajador", callback_data=f"wrk_rename:{worker.id}")],
    ]
    if worker.telegram_id < 0:
        rows.append([
            InlineKeyboardButton(text="🗑️ Eliminar trabajador de prueba", callback_data=f"wrk_del_dummy:{worker.id}")
        ])
    elif not is_env_superadmin:
        if worker.role == "admin":
            rows.append([InlineKeyboardButton(text="👤 Pasar a Trabajador", callback_data=f"wrk_role:{worker.id}:trabajador")])
        else:
            rows.append([InlineKeyboardButton(text="👑 Hacer Admin / Jefa", callback_data=f"wrk_role:{worker.id}:admin")])

    rows.append([InlineKeyboardButton(text="❌ Cerrar", callback_data="cancel_action")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_duplicate_decision_keyboard(pending_id: str) -> InlineKeyboardMarkup:
    """Inline choice when a potentially duplicated receipt/photo is uploaded."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🗑️ Descartar (Repetida)", callback_data=f"dup_drop:{pending_id}"),
                InlineKeyboardButton(text="✅ Guardar Igualmente", callback_data=f"dup_keep:{pending_id}"),
            ]
        ]
    )
