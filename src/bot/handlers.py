"""Telegram bot message handlers with RBAC, deduplication, and accounting workflows."""

from __future__ import annotations

import logging
import math
import os
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, FSInputFile, Message

from bot.commands import sync_user_chat_commands
from bot.keyboards import (
    ALL_MENU_BUTTONS,
    BTN_ASIGNAR_DINERO,
    BTN_AYUDA,
    BTN_BOLSILLO,
    BTN_CANCELAR,
    BTN_DEVOLVER,
    BTN_PANEL_ADMIN,
    BTN_PANEL_JEFA_LEGACY,
    BTN_RESUMEN,
    BTN_RESUMEN_LEGACY,
    BTN_SINCRONIZAR,
    BTN_SIN_NOTA,
    get_amount_inline_keyboard,
    get_boss_panel_inline_keyboard,
    get_cancel_keyboard,
    get_confirm_delete_ticket_keyboard,
    get_duplicate_decision_keyboard,
    get_main_keyboard,
    get_manage_tickets_keyboard,
    get_movements_delete_keyboard,
    get_note_keyboard,
    get_sync_inline_keyboard,
    get_ticket_actions_inline_keyboard,
    get_unassign_options_keyboard,
    get_worker_actions_keyboard,
    get_workers_inline_keyboard,
)
from db.models import User
from db.repository import Repository
from excel.writer import ensure_workbook, rebuild_excel_from_db, sync_pending_tickets
from ocr.dedup import compute_dhash, compute_sha256
from ocr.extractor import (
    ExtractedTicket,
    OllamaUnavailableError,
    VisionExtractor,
    is_suspicious_total,
    is_valid_ticket,
)
from ocr.preprocessor import resize_if_large

logger = logging.getLogger(__name__)

MAX_AMOUNT_EUR = 100000.0
MAX_NOTE_LENGTH = 200
DUPLICATE_CACHE_TTL_SECONDS = 3600.0


# ── FSM States for interactive flows ───────────────────────────────────────

class AssignMoneyStates(StatesGroup):
    waiting_for_amount = State()
    waiting_for_note = State()


class UnassignMoneyStates(StatesGroup):
    waiting_for_amount = State()
    waiting_for_note = State()


class ReturnMoneyStates(StatesGroup):
    waiting_for_amount = State()
    waiting_for_note = State()


class OwnPocketStates(StatesGroup):
    waiting_for_amount = State()
    waiting_for_note = State()


class CreateTestWorkerStates(StatesGroup):
    waiting_for_name = State()


class EditTicketTotalStates(StatesGroup):
    waiting_for_total = State()


class RenameWorkerStates(StatesGroup):
    waiting_for_name = State()


@dataclass
class _PendingDuplicate:
    """Temporarily holds a potentially duplicated upload awaiting user choice."""

    telegram_id: int
    user_id: int
    image_path: Path
    sha256: str
    dhash: str
    created_ts: float
    extracted: ExtractedTicket | None = None


def _sanitize_filename_segment(text: str, max_len: int = 24, fallback: str = "ticket") -> str:
    """Sanitize a string into an ASCII, Windows/SMB-safe filename segment."""
    if not text:
        return fallback
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned[:max_len] or fallback


def _format_photo_date_part(fecha: str) -> str:
    """Convert DD/MM/YYYY ticket date to YYYY-MM-DD, falling back to today."""
    raw = (fecha or "").strip()
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", raw)
    if m:
        day, month, year = m.groups()
        return f"{year}-{month}-{day}"
    return datetime.now().strftime("%Y-%m-%d")


def build_readable_ticket_image_path(
    images_dir: Path,
    worker_name: str,
    extracted: ExtractedTicket,
    unique_suffix: str | None = None,
) -> Path:
    """Build an organized, Windows/SMB-safe photo path under images_dir/YYYY-MM/."""
    month_folder = datetime.now().strftime("%Y-%m")
    date_part = _format_photo_date_part(extracted.fecha)
    worker_slug = _sanitize_filename_segment(worker_name, max_len=20, fallback="trabajador")
    empresa_slug = _sanitize_filename_segment(extracted.empresa, max_len=24, fallback="comercio")
    total_slug = f"{extracted.total:.2f}".replace(".", "-")
    suffix = _sanitize_filename_segment(unique_suffix or uuid.uuid4().hex[:6], max_len=12, fallback="img")

    filename = f"{date_part}_{worker_slug}_{empresa_slug}_{total_slug}EUR_{suffix}.jpg"
    target = (images_dir / month_folder / filename).resolve()
    base_resolved = images_dir.resolve()
    if not target.is_relative_to(base_resolved):
        raise ValueError("Invalid target image path outside images_dir")
    return target


def _store_ticket_image(
    temp_path: Path,
    images_dir: Path,
    worker_name: str,
    extracted: ExtractedTicket,
) -> Path:
    """Move processed receipt image into monthly readable export folder with 0644/0755 perms."""
    target_path = build_readable_ticket_image_path(images_dir, worker_name, extracted)
    try:
        os.chmod(images_dir, 0o755)
    except OSError:
        pass
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(target_path.parent, 0o755)
    except OSError:
        pass
    os.replace(temp_path, target_path)
    try:
        os.chmod(target_path, 0o644)
    except OSError:
        pass
    return target_path


# ── Helpers ───────────────────────────────────────────────────────────────

def is_admin(user: User | None, admin_telegram_ids: tuple[int, ...] = ()) -> bool:
    """Check if user has admin privileges (from .env superadmin list or bot-promoted admin)."""
    if user is None:
        return False
    if user.telegram_id in admin_telegram_ids:
        return True
    if admin_telegram_ids:
        return user.role == "admin" and user.admin_source == "bot"
    return user.role == "admin"


def get_user_menu(user: User | None, admin_telegram_ids: tuple[int, ...] = ()):
    """Return appropriate ReplyKeyboardMarkup for user's role."""
    return get_main_keyboard(is_admin=is_admin(user, admin_telegram_ids))


async def notify_admins(
    bot: Bot,
    repo: Repository,
    admin_telegram_ids: tuple[int, ...],
    text: str,
) -> None:
    """Send notification to all admins."""
    admin_users = repo.get_admins()
    target_ids = set(admin_telegram_ids)
    for u in admin_users:
        if u.telegram_id > 0:
            target_ids.add(u.telegram_id)
    for tid in target_ids:
        try:
            await bot.send_message(chat_id=tid, text=text)
        except Exception as e:
            logger.warning(f"Could not notify admin {tid}: {e}")


async def notify_worker(bot: Bot, worker_telegram_id: int, text: str) -> None:
    """Send notification to a worker if they have a real Telegram ID."""
    if worker_telegram_id <= 0:
        return
    try:
        await bot.send_message(chat_id=worker_telegram_id, text=text)
    except Exception as e:
        logger.warning(f"Could not notify worker {worker_telegram_id}: {e}")


def parse_euro_amount(text: str) -> float | None:
    """Safely parse user-entered money amount (e.g. '50', '50.50', '50,50 €')."""
    if not text:
        return None
    cleaned = text.replace("€", "").replace("eur", "").replace("EUR", "").strip()
    cleaned = cleaned.replace(",", ".")
    try:
        val = float(cleaned)
        if math.isnan(val) or math.isinf(val) or val <= 0 or val > MAX_AMOUNT_EUR:
            return None
        return round(val, 2)
    except ValueError:
        return None


def sanitize_note(raw_text: str) -> str:
    """Normalize and bound movement notes."""
    raw = (raw_text or "").strip()
    if raw in ("-", ".", BTN_SIN_NOTA, "Sin Nota", "sin nota"):
        return ""
    return raw[:MAX_NOTE_LENGTH]


async def _clear_state(state: FSMContext | None) -> None:
    """Safely clear FSM state when provided."""
    if state is not None:
        await state.clear()


def _require_admin_sync(
    telegram_id: int,
    repo: Repository,
    admin_telegram_ids: tuple[int, ...],
) -> User | None:
    """Return User if admin, else None (auto-demoting stale .env/legacy admin rows)."""
    user = repo.get_user_by_telegram_id(telegram_id)
    if user is None:
        return None
    if is_admin(user, admin_telegram_ids):
        return user
    if admin_telegram_ids and user.role == "admin" and user.admin_source != "bot" and user.id is not None:
        repo.set_user_role(user.id, "trabajador", admin_source="")
        user.role = "trabajador"
        user.admin_source = ""
    return None


# ── Router factory ────────────────────────────────────────────────────────

def create_router(
    repo: Repository,
    extractor: VisionExtractor,
    bot: Bot,
    images_dir: Path,
    excel_path: Path,
    admin_telegram_ids: tuple[int, ...] = (),
) -> Router:
    """Create and configure the bot router with dependencies."""
    router = Router()
    pending_duplicates: dict[str, _PendingDuplicate] = {}

    def _prune_pending_duplicates() -> None:
        """Delete stale unresolved duplicate entries and orphan images older than TTL."""
        now = time.monotonic()
        expired = [k for k, v in pending_duplicates.items() if now - v.created_ts > DUPLICATE_CACHE_TTL_SECONDS]
        for key in expired:
            entry = pending_duplicates.pop(key, None)
            if entry is not None:
                entry.image_path.unlink(missing_ok=True)

    async def _require_admin_msg(message: Message, state: FSMContext | None = None) -> User | None:
        admin_user = _require_admin_sync(message.from_user.id, repo, admin_telegram_ids)
        if admin_user is None:
            await _clear_state(state)
            user = repo.get_user_by_telegram_id(message.from_user.id)
            reply_markup = get_user_menu(user, admin_telegram_ids) if user is not None else None
            await message.answer(
                "⛔ Esta opción es exclusiva para Administradores.",
                reply_markup=reply_markup,
            )
            return None
        return admin_user

    async def _require_admin_cb(callback: CallbackQuery, state: FSMContext | None = None) -> User | None:
        admin_user = _require_admin_sync(callback.from_user.id, repo, admin_telegram_ids)
        if admin_user is None:
            await _clear_state(state)
            await callback.answer("⛔ Esta opción es exclusiva para Administradores.", show_alert=True)
            return None
        return admin_user

    async def _advance_amount_to_note(
        target_message: Message,
        state: FSMContext,
        next_state: State,
        amount: float,
        prompt_text: str,
    ) -> None:
        await state.update_data(amount=amount)
        await state.set_state(next_state)
        await target_message.answer(prompt_text, reply_markup=get_note_keyboard())

    async def _persist_and_confirm_ticket(
        reply_target: Message,
        user: User,
        image_path: Path,
        extracted: ExtractedTicket,
        sha256: str,
        dhash: str,
    ) -> None:
        """Save extracted ticket to SQLite, rename photo cleanly, sync Excel, and reply."""
        final_image_path = image_path
        if image_path.exists():
            final_image_path = _store_ticket_image(image_path, images_dir, user.name, extracted)

        saved_ticket = repo.create_ticket(
            user_id=user.id,
            image_path=str(final_image_path),
            empresa=extracted.empresa,
            nif=extracted.nif,
            descripcion=extracted.descripcion,
            total=extracted.total,
            fecha=extracted.fecha,
            synced_to_excel=False,
            image_sha256=sha256,
            image_dhash=dhash,
        )

        _synced_count, sync_ok = sync_pending_tickets(excel_path, repo)
        bal = repo.get_user_balance(user.id)
        user_is_admin = is_admin(user, admin_telegram_ids)

        nif_line = f"🪪 {extracted.nif}\n" if extracted.nif else ""
        notes: list[str] = []
        if not sync_ok:
            if user_is_admin:
                notes.append(
                    "⚠️ <i>Nota: El archivo Excel está abierto en el equipo. "
                    "El ticket se guardó en la base de datos y se sincronizará automáticamente cuando lo cierres.</i>"
                )
            else:
                await notify_admins(
                    bot,
                    repo,
                    admin_telegram_ids,
                    (
                        f"⚠️ <b>Excel bloqueado al guardar ticket de {user.name}</b>\n"
                        "El ticket quedó guardado en la base de datos, pero <code>tickets.xlsx</code> estaba abierto. "
                        "Cerralo y pulsá 🔄 Sincronizar Excel."
                    ),
                )
        if is_suspicious_total(extracted.total):
            notes.append(
                f"⚠️ <i>Aviso: El importe ({extracted.total:.2f} €) parece inusualmente alto. "
                "Verificá si coincide con el ticket.</i>"
            )

        extra_notes = ("\n\n" + "\n".join(notes)) if notes else ""
        if not sync_ok and user_is_admin:
            reply_markup = get_sync_inline_keyboard()
        elif saved_ticket.id is not None:
            reply_markup = get_ticket_actions_inline_keyboard(saved_ticket.id)
        else:
            reply_markup = get_user_menu(user, admin_telegram_ids)

        await reply_target.answer(
            f"✅ <b>Ticket procesado con éxito</b>\n\n"
            f"🏢 {extracted.empresa}\n"
            f"{nif_line}"
            f"📝 {extracted.descripcion}\n"
            f"💰 {extracted.total:.2f} €\n"
            f"📅 {extracted.fecha}\n\n"
            f"📊 <b>Tu saldo actualizado:</b> {bal.status_label}"
            f"{extra_notes}",
            reply_markup=reply_markup,
        )

    # ── Start & Cancel (always interrupt any active FSM state) ────────────

    @router.message(StateFilter("*"), Command("start"))
    async def handle_start(message: Message, state: FSMContext | None = None) -> None:
        """Handle /start command — register user if new, or show welcome back."""
        await _clear_state(state)
        telegram_id = message.from_user.id
        existing = repo.get_user_by_telegram_id(telegram_id)

        if existing:
            if telegram_id in admin_telegram_ids:
                if (existing.role != "admin" or existing.admin_source != "env") and existing.id is not None:
                    repo.set_user_role(existing.id, "admin", admin_source="env")
                    existing.role = "admin"
                    existing.admin_source = "env"
            elif existing.role == "admin" and existing.admin_source != "bot" and existing.id is not None:
                repo.set_user_role(existing.id, "trabajador", admin_source="")
                existing.role = "trabajador"
                existing.admin_source = ""

            user_is_admin = is_admin(existing, admin_telegram_ids)
            await sync_user_chat_commands(bot, telegram_id, is_admin=user_is_admin)
            role_desc = "Admin 👑" if user_is_admin else "Trabajador 👤"
            await message.answer(
                f"👋 ¡Hola de nuevo, {existing.name}!\n"
                f"Rol actual: <b>{role_desc}</b>\n\n"
                "📸 Mandame fotos de tus tickets o usá los botones de abajo:",
                reply_markup=get_user_menu(existing, admin_telegram_ids),
            )
            return

        await message.answer(
            "👋 ¡Hola! Soy el bot de tickets y control de caja.\n\n"
            "¿Cómo te llamás? Escribime tu nombre completo (ej: Juan Pérez)."
        )

    @router.message(StateFilter("*"), Command("cancelar"))
    @router.message(StateFilter("*"), F.text == BTN_CANCELAR)
    async def handle_cancel_text(message: Message, state: FSMContext | None = None) -> None:
        """Cancel any active FSM dialog via reply button or command."""
        await _clear_state(state)
        user = repo.get_user_by_telegram_id(message.from_user.id)
        await message.answer(
            "❌ Acción cancelada. Volvemos al menú principal.",
            reply_markup=get_user_menu(user, admin_telegram_ids),
        )

    @router.callback_query(F.data == "cancel_action")
    async def handle_cancel_callback(callback: CallbackQuery, state: FSMContext | None = None) -> None:
        """Cancel any active FSM dialog via inline button."""
        await _clear_state(state)
        try:
            if callback.message:
                await callback.message.delete()
        except Exception:
            pass
        user = repo.get_user_by_telegram_id(callback.from_user.id)
        if callback.message:
            await callback.message.answer(
                "❌ Acción cancelada.",
                reply_markup=get_user_menu(user, admin_telegram_ids),
            )
        await callback.answer()

    # ── Help (interrupts FSM cleanly) ─────────────────────────────────────

    @router.message(StateFilter("*"), Command("ayuda"))
    @router.message(StateFilter("*"), F.text == BTN_AYUDA)
    async def handle_help(message: Message, state: FSMContext | None = None) -> None:
        """Handle /ayuda command or Ayuda button — show role-tailored guide."""
        await _clear_state(state)
        user = repo.get_user_by_telegram_id(message.from_user.id)
        admin_mode = is_admin(user, admin_telegram_ids)

        if admin_mode:
            text = (
                "👑 <b>Panel de Administración</b>\n\n"
                "• <b>👑 Panel Admin:</b> Ver balances de todos los trabajadores, descargar Excel, desasignar/anular movimientos o gestionar tickets y usuarios.\n"
                "• <b>💵 Asignar Dinero:</b> Entregar efectivo con tarjetas rápidas o importe manual.\n"
                "• <b>📊 Mi Cuenta / Saldo:</b> Ver tu balance individual.\n"
                "• <b>💸 Devolver Sobrante:</b> Registrar devolución de efectivo.\n"
                "• <b>🔄 Sincronizar Excel:</b> Reconstruir el archivo Excel íntegro desde la base de datos.\n"
                "• 📸 Podés mandarme fotos de tickets en cualquier momento (avisa si alguna foto está repetida)."
            )
        else:
            text = (
                "👤 <b>Guía del Trabajador</b>\n\n"
                "• 📸 <b>Enviar Ticket:</b> Mandá una foto clara de cualquier ticket/factura. El Excel se actualiza automáticamente y podrás corregir el total o anularlo justo debajo si hizo falta.\n"
                "• <b>📊 Mi Cuenta / Saldo:</b> Consultá cuánto dinero te asignó la empresa, cuánto gastaste y tu saldo.\n"
                "• <b>💸 Devolver Sobrante:</b> Registrá cuando devuelvas efectivo a caja.\n"
                "• <b>👛 Puse de mi Bolsillo:</b> Dejá constancia si pusiste dinero propio hoy."
            )

        await message.answer(text, reply_markup=get_user_menu(user, admin_telegram_ids))

    # ── Synchronize & Download Excel ──────────────────────────────────────

    @router.message(StateFilter("*"), Command("sincronizar", "sync"))
    @router.message(StateFilter("*"), F.text == BTN_SINCRONIZAR)
    async def handle_sync(message: Message, state: FSMContext | None = None) -> None:
        """Handle /sincronizar command or button — full rebuild of Excel from SQLite (Admin only)."""
        admin_user = await _require_admin_msg(message, state)
        if admin_user is None:
            return
        await _clear_state(state)

        synced_count, sync_ok = rebuild_excel_from_db(excel_path, repo)
        if not sync_ok:
            await message.answer(
                "⚠️ El archivo Excel sigue abierto o bloqueado en el equipo.\n\n"
                "Por favor cerralo y volvé a tocar 🔄 Sincronizar Excel.",
                reply_markup=get_sync_inline_keyboard(),
            )
            return

        if synced_count > 0:
            await message.answer(
                f"✅ ¡Sincronización completa!\n"
                f"Se reconstruyó el libro Excel completo desde la base de datos ({synced_count} registro(s)).",
                reply_markup=get_user_menu(admin_user, admin_telegram_ids),
            )
        else:
            await message.answer(
                "✅ El archivo Excel ya está al día (sin registros en la base de datos).",
                reply_markup=get_user_menu(admin_user, admin_telegram_ids),
            )

    @router.callback_query(F.data == "do_sync")
    async def handle_callback_sync(callback: CallbackQuery) -> None:
        """Handle inline button press for immediate sync (Admin only)."""
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        synced_count, sync_ok = sync_pending_tickets(excel_path, repo)
        if not sync_ok:
            await callback.answer(
                "⚠️ El archivo Excel sigue abierto en el equipo.\nCerralo y volvé a tocar.",
                show_alert=True,
            )
        else:
            msg = (
                f"✅ ¡Listo! Se volcaron {synced_count} tickets al Excel."
                if synced_count > 0
                else "✅ El archivo Excel ya está 100% al día."
            )
            await callback.answer(msg, show_alert=True)
            try:
                if callback.message:
                    await callback.message.edit_reply_markup(reply_markup=None)
            except Exception:
                pass

    @router.message(StateFilter("*"), Command("excel"))
    async def handle_excel_command(message: Message, state: FSMContext | None = None) -> None:
        """Send up-to-date Excel workbook directly in Telegram (Admin only)."""
        admin_user = await _require_admin_msg(message, state)
        if admin_user is None:
            return
        await _clear_state(state)
        sync_pending_tickets(excel_path, repo)
        if not excel_path.exists():
            ensure_workbook(excel_path)
        await message.answer_document(
            document=FSInputFile(excel_path, filename=excel_path.name),
            caption="📊 Aquí tenés el archivo Excel de contabilidad y tickets actualizado.",
        )

    @router.callback_query(F.data == "excel_download")
    async def handle_excel_download(callback: CallbackQuery) -> None:
        """Send up-to-date Excel workbook from Boss Panel inline button (Admin only)."""
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        sync_pending_tickets(excel_path, repo)
        if not excel_path.exists():
            ensure_workbook(excel_path)
        if callback.message:
            await callback.message.answer_document(
                document=FSInputFile(excel_path, filename=excel_path.name),
                caption="📊 Excel de contabilidad y tickets actualizado.",
            )
        await callback.answer("📥 Excel enviado")

    # ── Mi Cuenta / Saldo ─────────────────────────────────────────────────

    @router.message(StateFilter("*"), Command("resumen", "saldo"))
    @router.message(StateFilter("*"), F.text == BTN_RESUMEN)
    @router.message(StateFilter("*"), F.text == BTN_RESUMEN_LEGACY)
    async def handle_summary(message: Message, state: FSMContext | None = None) -> None:
        """Show user's comprehensive accounting balance and last tickets."""
        await _clear_state(state)
        user = repo.get_user_by_telegram_id(message.from_user.id)
        if user is None:
            await message.answer("Primero escribí /start para registrarte.")
            return

        bal = repo.get_user_balance(user.id)
        tickets = repo.get_tickets_by_user(user.id)

        lines = [
            f"👤 <b>Cuenta de {user.name} ({bal.month_label})</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]
        if abs(bal.previous_balance) >= 0.01:
            sign = "+" if bal.previous_balance > 0 else ""
            lines.append(f"📌 <i>Saldo mes anterior:</i> {sign}{bal.previous_balance:.2f} €")

        lines.extend([
            f"💵 <b>Entregado este mes:</b> {bal.month_assigned:.2f} €",
            f"🧾 <b>Gastado este mes ({bal.month_tickets_count} ticket{'s' if bal.month_tickets_count != 1 else ''}):</b> {bal.month_spent:.2f} €",
            f"💸 <b>Devuelto este mes:</b> {bal.month_returned:.2f} €",
        ])
        if bal.total_own_pocket > 0:
            lines.append(f"👛 <i>Fondo Propio Decl.:</i> {bal.total_own_pocket:.2f} €")

        lines.extend([
            "──────────────────────",
            f"💰 <b>SALDO NETO ACTUAL:</b> {bal.net_balance:.2f} €",
            f"📊 <b>Estado:</b> {bal.status_label}\n",
        ])

        if abs(bal.net_balance) < 0.01:
            lines.append("🟢 <i>Estás al día. A paz y salvo con la empresa.</i>")
        elif bal.net_balance > 0:
            lines.append(
                f"🟡 <i>Tenés <b>{bal.net_balance:.2f} €</b> de la empresa pendientes de justificar con tickets o devolver.</i>"
            )
        else:
            lines.append(
                f"🔴 <i>La empresa te debe reembolsar <b>{abs(bal.net_balance):.2f} €</b> de tickets pagados con tu dinero.</i>"
            )

        if tickets:
            lines.append("\n<b>Últimos tickets subidos:</b>")
            for t in tickets[:5]:
                lines.append(f"• {t.fecha} — {t.empresa}: {t.total:.2f} €")

        await message.answer("\n".join(lines), reply_markup=get_user_menu(user, admin_telegram_ids))

    # ── Panel Admin (Admin Only) ──────────────────────────────────────────

    @router.message(StateFilter("*"), Command("panel"))
    @router.message(StateFilter("*"), F.text.in_({BTN_PANEL_ADMIN, BTN_PANEL_JEFA_LEGACY}))
    async def handle_boss_panel(message: Message, state: FSMContext | None = None) -> None:
        """Show admin control panel with balances and quick management actions."""
        admin_user = await _require_admin_msg(message, state)
        if admin_user is None:
            return
        await _clear_state(state)

        balances = repo.get_all_balances()
        if not balances:
            await message.answer("Aún no hay trabajadores registrados en el sistema.")
            return

        month_label = balances[0].month_label if balances else ""
        lines = [
            f"👑 <b>Panel Admin ({month_label})</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        total_asig_mes = 0.0
        total_spent_mes = 0.0
        total_ret_mes = 0.0
        total_net_global = 0.0

        for b in balances:
            total_asig_mes += b.month_assigned
            total_spent_mes += b.month_spent
            total_ret_mes += b.month_returned
            total_net_global += b.net_balance

            role_tag = " (Admin)" if b.role == "admin" else ""
            lines.append(
                f"👤 <b>{b.name}</b>{role_tag}\n"
                f"   💵 Entregado este mes: {b.month_assigned:.2f} € | 🧾 Gastado: {b.month_spent:.2f} €\n"
                f"   💰 Saldo Neto Actual: <b>{b.net_balance:.2f} €</b>\n"
                f"   📊 {b.status_label}\n"
            )

        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📈 <b>Totales de la Empresa ({month_label}):</b>",
            f"• Total Entregado este mes: <b>{total_asig_mes:.2f} €</b>",
            f"• Total Gastado este mes: <b>{total_spent_mes:.2f} €</b>",
            f"• Total Devuelto este mes: <b>{total_ret_mes:.2f} €</b>",
            f"• Balance Global Pendiente: <b>{total_net_global:.2f} €</b>",
            "\n💡 <i>Usá las acciones rápidas debajo para descargar el Excel o corregir movimientos/tickets.</i>",
        ])

        await message.answer("\n".join(lines), reply_markup=get_boss_panel_inline_keyboard())

    # ── Asignar Dinero (Admin Only Flow) ──────────────────────────────────

    @router.message(StateFilter("*"), Command("asignar"))
    @router.message(StateFilter("*"), F.text == BTN_ASIGNAR_DINERO)
    async def handle_assign_money_start(message: Message, state: FSMContext | None = None) -> None:
        """Start assign money flow by picking a worker."""
        admin_user = await _require_admin_msg(message, state)
        if admin_user is None:
            return
        await _clear_state(state)

        workers = repo.get_all_users()
        if not workers:
            await message.answer("No hay trabajadores registrados aún.")
            return

        await message.answer(
            "💵 <b>Asignar Dinero a un Trabajador</b>\n\n"
            "Elegí a quién le vas a entregar efectivo:",
            reply_markup=get_workers_inline_keyboard(workers),
        )

    @router.callback_query(F.data.startswith("asig_worker:"))
    async def handle_assign_worker_chosen(callback: CallbackQuery, state: FSMContext) -> None:
        """Worker selected from inline keyboard."""
        admin_user = await _require_admin_cb(callback, state)
        if admin_user is None:
            return

        worker_id = int(callback.data.split(":")[1])
        worker = repo.get_user_by_id(worker_id)
        if worker is None:
            await callback.answer("Trabajador no encontrado.", show_alert=True)
            return

        await state.update_data(target_worker_id=worker.id, target_worker_name=worker.name)
        await state.set_state(AssignMoneyStates.waiting_for_amount)

        await callback.message.edit_text(
            f"💵 Asignar dinero a <b>{worker.name}</b>\n\n"
            "Tocá un importe rápido o escribí la cantidad en euros en el chat (ej: 75.50):",
            reply_markup=get_amount_inline_keyboard("asig_amt"),
        )
        await callback.message.answer(
            "Podés pulsar el botón de abajo para cancelar en cualquier momento:",
            reply_markup=get_cancel_keyboard(),
        )
        await callback.answer()

    @router.callback_query(AssignMoneyStates.waiting_for_amount, F.data.startswith("asig_amt:"))
    async def handle_assign_amount_card(callback: CallbackQuery, state: FSMContext) -> None:
        """Handle quick amount card press during Assign Money flow."""
        admin_user = await _require_admin_cb(callback, state)
        if admin_user is None:
            return

        amount = parse_euro_amount(callback.data.split(":", 1)[1])
        if amount is None:
            await callback.answer("Importe no válido.", show_alert=True)
            return

        data = await state.get_data()
        worker_name = data.get("target_worker_name", "Trabajador")
        try:
            if callback.message:
                await callback.message.edit_text(f"💵 Asignar a <b>{worker_name}</b>: <b>{amount:.2f} €</b>")
        except Exception:
            pass

        await _advance_amount_to_note(
            callback.message,
            state,
            AssignMoneyStates.waiting_for_note,
            amount,
            f"Importe: <b>{amount:.2f} €</b>\n\nIngresá un concepto o motivo (ej: <code>Gasolina semana</code>) o tocá el botón de abajo:",
        )
        await callback.answer()

    @router.callback_query(F.data == "asig_create_dummy")
    async def handle_create_dummy_start(callback: CallbackQuery, state: FSMContext) -> None:
        """Allow admin to create a test worker for instant multi-user simulation without a 2nd phone."""
        admin_user = await _require_admin_cb(callback, state)
        if admin_user is None:
            return

        await state.set_state(CreateTestWorkerStates.waiting_for_name)
        await callback.message.edit_text(
            "➕ <b>Crear Trabajador de Prueba</b>\n\n"
            "Escribí el nombre del trabajador de prueba (ej: <code>Carlos Chofer</code>):"
        )
        await callback.message.answer(
            "Podés pulsar el botón de abajo para cancelar:",
            reply_markup=get_cancel_keyboard(),
        )
        await callback.answer()

    @router.message(CreateTestWorkerStates.waiting_for_name)
    async def handle_create_dummy_name(message: Message, state: FSMContext) -> None:
        admin_user = await _require_admin_msg(message, state)
        if admin_user is None:
            return

        name = message.text.strip()
        if len(name) < 2 or len(name) > 100:
            await message.answer("El nombre debe tener entre 2 y 100 caracteres. Escribí otro nombre:")
            return

        dummy = repo.create_dummy_worker(name)
        await state.clear()
        await message.answer(
            f"✅ Se creó el trabajador de prueba <b>{dummy.name}</b>.\n"
            "Ahora ya aparece en el panel y podés asignarle dinero para probar.",
            reply_markup=get_user_menu(admin_user, admin_telegram_ids),
        )

    @router.message(AssignMoneyStates.waiting_for_amount)
    async def handle_assign_amount(message: Message, state: FSMContext) -> None:
        admin_user = await _require_admin_msg(message, state)
        if admin_user is None:
            return

        amount = parse_euro_amount(message.text)
        if amount is None:
            await message.answer(
                "⚠️ Importe no válido. Por favor ingresá un número positivo en euros (ej: 50 o 50.50):",
                reply_markup=get_cancel_keyboard(),
            )
            return

        await _advance_amount_to_note(
            message,
            state,
            AssignMoneyStates.waiting_for_note,
            amount,
            f"Importe: <b>{amount:.2f} €</b>\n\nIngresá un concepto o motivo (ej: <code>Gasolina semana</code>) o tocá el botón de abajo:",
        )

    @router.message(AssignMoneyStates.waiting_for_note)
    async def handle_assign_note(message: Message, state: FSMContext) -> None:
        admin_user = await _require_admin_msg(message, state)
        if admin_user is None:
            return

        data = await state.get_data()
        worker_id = data["target_worker_id"]
        worker_name = data["target_worker_name"]
        amount = data["amount"]
        note = sanitize_note(message.text)

        admin_id = admin_user.id if admin_user.id is not None else worker_id
        repo.create_cash_movement(
            user_id=worker_id,
            created_by_id=admin_id,
            movement_type="ASIGNACION",
            amount=amount,
            note=note,
        )

        sync_pending_tickets(excel_path, repo)
        await state.clear()

        await message.answer(
            f"✅ <b>Asignación registrada con éxito</b>\n\n"
            f"👤 Trabajador: <b>{worker_name}</b>\n"
            f"💵 Importe: <b>{amount:.2f} €</b>\n"
            f"📝 Nota: {note or 'Sin nota'}",
            reply_markup=get_user_menu(admin_user, admin_telegram_ids),
        )

        worker = repo.get_user_by_id(worker_id)
        if worker and worker.telegram_id > 0:
            notify_text = (
                f"💵 <b>Administración te ha entregado {amount:.2f} € en efectivo</b>\n"
                f"📝 Motivo: {note or 'Fondo de caja'}\n\n"
                "Podés ver tu saldo actualizado pulsando <b>📊 Mi Cuenta / Saldo</b>."
            )
            await notify_worker(bot, worker.telegram_id, notify_text)

    # ── Desasignar Dinero / Anular Movimiento (Admin Only) ────────────────

    @router.callback_query(F.data == "admin_unassign_menu")
    async def handle_admin_unassign_menu(callback: CallbackQuery) -> None:
        """Show worker list to unassign or void cash movements."""
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        workers = repo.get_all_users()
        if not workers:
            await callback.answer("No hay trabajadores registrados.", show_alert=True)
            return
        await callback.message.answer(
            "↩️ <b>Desasignar Dinero o Anular Movimiento</b>\n\nElegí el trabajador:",
            reply_markup=get_workers_inline_keyboard(workers, callback_prefix="unasig_worker", include_create_dummy=False),
        )
        await callback.answer()

    @router.callback_query(F.data.startswith("unasig_worker:"))
    async def handle_unassign_worker_chosen(callback: CallbackQuery) -> None:
        """Choose between custom unassignment or voiding a specific movement."""
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        worker_id = int(callback.data.split(":")[1])
        worker = repo.get_user_by_id(worker_id)
        if worker is None:
            await callback.answer("Trabajador no encontrado.", show_alert=True)
            return

        bal = repo.get_user_balance(worker.id)
        await callback.message.edit_text(
            f"↩️ <b>Ajustes de Caja — {worker.name}</b>\n"
            f"Saldo neto actual: <b>{bal.status_label}</b>\n\n"
            "• <b>➖ Desasignar importe:</b> Descuenta efectivo asignado dejando constancia en Movimientos Caja.\n"
            "• <b>🗑️ Anular un movimiento:</b> Borra por completo un registro erróneo (ej: pusiste 500 € en vez de 50 €).",
            reply_markup=get_unassign_options_keyboard(worker.id),
        )
        await callback.answer()

    @router.callback_query(F.data.startswith("unassign_custom:"))
    async def handle_unassign_custom(callback: CallbackQuery, state: FSMContext) -> None:
        """Start partial cash unassignment with quick amount cards."""
        admin_user = await _require_admin_cb(callback, state)
        if admin_user is None:
            return
        worker_id = int(callback.data.split(":")[1])
        worker = repo.get_user_by_id(worker_id)
        if worker is None:
            await callback.answer("Trabajador no encontrado.", show_alert=True)
            return

        bal = repo.get_user_balance(worker.id)
        await state.update_data(target_worker_id=worker.id, target_worker_name=worker.name)
        await state.set_state(UnassignMoneyStates.waiting_for_amount)

        extra = (f"↩️ Retirar todo ({bal.net_balance:.2f} €)", f"unassign_amt:{bal.net_balance:.2f}") if bal.net_balance > 0 else None
        await callback.message.edit_text(
            f"➖ Desasignar dinero de <b>{worker.name}</b>\n"
            f"Saldo actual: <b>{bal.net_balance:.2f} €</b>\n\n"
            "Tocá un importe rápido o escribí cuánto querés desasignar en euros:",
            reply_markup=get_amount_inline_keyboard("unassign_amt", amounts=(20, 50, 100), extra_top_button=extra),
        )
        await callback.answer()

    @router.callback_query(UnassignMoneyStates.waiting_for_amount, F.data.startswith("unassign_amt:"))
    async def handle_unassign_amount_card(callback: CallbackQuery, state: FSMContext) -> None:
        admin_user = await _require_admin_cb(callback, state)
        if admin_user is None:
            return
        amount = parse_euro_amount(callback.data.split(":", 1)[1])
        if amount is None:
            await callback.answer("Importe no válido.", show_alert=True)
            return
        data = await state.get_data()
        worker_name = data.get("target_worker_name", "Trabajador")
        try:
            await callback.message.edit_text(f"➖ Desasignar de <b>{worker_name}</b>: <b>{amount:.2f} €</b>")
        except Exception:
            pass
        await _advance_amount_to_note(
            callback.message,
            state,
            UnassignMoneyStates.waiting_for_note,
            amount,
            f"Importe a desasignar: <b>{amount:.2f} €</b>\n\nIngresá un motivo (ej: <code>Ajuste de caja</code>) o tocá el botón:",
        )
        await callback.answer()

    @router.message(UnassignMoneyStates.waiting_for_amount)
    async def handle_unassign_amount(message: Message, state: FSMContext) -> None:
        admin_user = await _require_admin_msg(message, state)
        if admin_user is None:
            return
        amount = parse_euro_amount(message.text)
        if amount is None:
            await message.answer(
                "⚠️ Importe no válido. Ingresá un número positivo en euros (ej: 50):",
                reply_markup=get_cancel_keyboard(),
            )
            return
        await _advance_amount_to_note(
            message,
            state,
            UnassignMoneyStates.waiting_for_note,
            amount,
            f"Importe a desasignar: <b>{amount:.2f} €</b>\n\nIngresá un motivo o tocá el botón:",
        )

    @router.message(UnassignMoneyStates.waiting_for_note)
    async def handle_unassign_note(message: Message, state: FSMContext) -> None:
        admin_user = await _require_admin_msg(message, state)
        if admin_user is None:
            return
        data = await state.get_data()
        worker_id = data["target_worker_id"]
        worker_name = data["target_worker_name"]
        amount = data["amount"]
        note = sanitize_note(message.text)

        admin_id = admin_user.id if admin_user.id is not None else worker_id
        repo.create_cash_movement(
            user_id=worker_id,
            created_by_id=admin_id,
            movement_type="DESASIGNACION",
            amount=amount,
            note=note,
        )
        sync_pending_tickets(excel_path, repo)
        bal = repo.get_user_balance(worker_id)
        await state.clear()

        await message.answer(
            f"✅ <b>Desasignación registrada con éxito</b>\n\n"
            f"👤 Trabajador: <b>{worker_name}</b>\n"
            f"↩️ Importe retirado: <b>{amount:.2f} €</b>\n"
            f"📝 Nota: {note or 'Sin nota'}\n"
            f"📊 Nuevo estado: <b>{bal.status_label}</b>",
            reply_markup=get_user_menu(admin_user, admin_telegram_ids),
        )
        worker = repo.get_user_by_id(worker_id)
        if worker and worker.telegram_id > 0:
            await notify_worker(
                bot,
                worker.telegram_id,
                f"↩️ <b>Administración ha desasignado / ajustado {amount:.2f} € de tu cuenta</b>\n"
                f"📝 Motivo: {note or 'Ajuste de asignación'}\n"
                f"📊 Tu nuevo estado: {bal.status_label}",
            )

    @router.callback_query(F.data.startswith("unassign_void_list:"))
    async def handle_unassign_void_list(callback: CallbackQuery) -> None:
        """List worker's recent movements to void an erroneous entry."""
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        worker_id = int(callback.data.split(":")[1])
        movements = repo.get_recent_movements(user_id=worker_id, limit=6)
        if not movements:
            await callback.answer("No hay movimientos recientes para este trabajador.", show_alert=True)
            return
        worker_name = movements[0][1]
        await callback.message.edit_text(
            f"🗑️ <b>Anular Movimiento — {worker_name}</b>\n\n"
            "Tocá el movimiento que querés eliminar de la BD y del Excel:",
            reply_markup=get_movements_delete_keyboard(movements),
        )
        await callback.answer()

    @router.callback_query(F.data.startswith("mov_del:"))
    async def handle_movement_delete(callback: CallbackQuery) -> None:
        """Delete a mistaken cash movement and rebuild Excel cleanly."""
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        mov_id = int(callback.data.split(":")[1])
        existing = repo.get_movement_by_id(mov_id)
        if existing is None:
            await callback.answer("Ese movimiento ya no existe.", show_alert=True)
            return

        mov, worker_name, _creator_name = existing
        deleted = repo.delete_cash_movement(mov_id)
        if deleted is None:
            await callback.answer("No se pudo eliminar el movimiento.", show_alert=True)
            return

        rebuild_excel_from_db(excel_path, repo)
        bal = repo.get_user_balance(mov.user_id)

        await callback.message.edit_text(
            f"🗑️ <b>Movimiento anulado con éxito</b>\n\n"
            f"👤 Trabajador: <b>{worker_name}</b>\n"
            f"🏷️ Tipo: <code>{mov.movement_type}</code> — <b>{mov.amount:.2f} €</b>\n"
            f"📊 Nuevo estado: <b>{bal.status_label}</b>\n\n"
            "<i>El Excel ha sido recalculado y limpiado automáticamente.</i>"
        )
        worker = repo.get_user_by_id(mov.user_id)
        if worker and worker.telegram_id > 0:
            await notify_worker(
                bot,
                worker.telegram_id,
                f"ℹ️ <b>Administración anuló un movimiento erróneo ({mov.movement_type}: {mov.amount:.2f} €)</b>\n"
                f"📊 Tu estado actualizado: {bal.status_label}",
            )
        await callback.answer("Movimiento anulado")

    # ── Devolver Sobrante Flow ────────────────────────────────────────────

    @router.message(StateFilter("*"), Command("devolver"))
    @router.message(StateFilter("*"), F.text == BTN_DEVOLVER)
    async def handle_return_money_start(message: Message, state: FSMContext | None = None) -> None:
        """Start return leftover cash flow."""
        await _clear_state(state)
        user = repo.get_user_by_telegram_id(message.from_user.id)
        if user is None:
            await message.answer("Primero escribí /start para registrarte.")
            return

        bal = repo.get_user_balance(user.id)
        if state is not None:
            await state.set_state(ReturnMoneyStates.waiting_for_amount)

        extra = (f"💸 Devolver todo ({bal.net_balance:.2f} €)", f"ret_amt:{bal.net_balance:.2f}") if bal.net_balance > 0 else None
        await message.answer(
            f"💸 <b>Devolver Sobrante a Caja / Empresa</b>\n"
            f"Tu saldo pendiente actual: <b>{bal.net_balance:.2f} €</b>\n\n"
            "Tocá un importe rápido o escribí cuánto vas a devolver en euros:",
            reply_markup=get_amount_inline_keyboard("ret_amt", amounts=(20, 50, 100), extra_top_button=extra),
        )

    @router.callback_query(ReturnMoneyStates.waiting_for_amount, F.data.startswith("ret_amt:"))
    async def handle_return_amount_card(callback: CallbackQuery, state: FSMContext) -> None:
        amount = parse_euro_amount(callback.data.split(":", 1)[1])
        if amount is None:
            await callback.answer("Importe no válido.", show_alert=True)
            return
        try:
            await callback.message.edit_text(f"💸 Devolver sobrante: <b>{amount:.2f} €</b>")
        except Exception:
            pass
        await _advance_amount_to_note(
            callback.message,
            state,
            ReturnMoneyStates.waiting_for_note,
            amount,
            f"Importe a devolver: <b>{amount:.2f} €</b>\n\nIngresá un concepto o nota (ej: <code>Sobrante viaje Madrid</code>) o tocá el botón de abajo:",
        )
        await callback.answer()

    @router.message(ReturnMoneyStates.waiting_for_amount)
    async def handle_return_amount(message: Message, state: FSMContext) -> None:
        amount = parse_euro_amount(message.text)
        if amount is None:
            await message.answer(
                "⚠️ Importe no válido. Por favor ingresá un número positivo en euros (ej: 20 o 20.50):",
                reply_markup=get_cancel_keyboard(),
            )
            return
        await _advance_amount_to_note(
            message,
            state,
            ReturnMoneyStates.waiting_for_note,
            amount,
            f"Importe a devolver: <b>{amount:.2f} €</b>\n\nIngresá un concepto o nota (ej: <code>Sobrante viaje Madrid</code>) o tocá el botón de abajo:",
        )

    @router.message(ReturnMoneyStates.waiting_for_note)
    async def handle_return_note(message: Message, state: FSMContext) -> None:
        data = await state.get_data()
        amount = data["amount"]
        note = sanitize_note(message.text)

        user = repo.get_user_by_telegram_id(message.from_user.id)
        if user is None:
            await state.clear()
            return

        repo.create_cash_movement(
            user_id=user.id,
            created_by_id=user.id,
            movement_type="DEVOLUCION",
            amount=amount,
            note=note,
        )

        sync_pending_tickets(excel_path, repo)
        bal = repo.get_user_balance(user.id)
        await state.clear()

        await message.answer(
            f"✅ <b>Devolución registrada con éxito</b>\n\n"
            f"💵 Devolviste: <b>{amount:.2f} €</b>\n"
            f"📝 Concepto: {note or 'Sobrante'}\n"
            f"💰 Tu nuevo saldo: <b>{bal.status_label}</b>",
            reply_markup=get_user_menu(user, admin_telegram_ids),
        )

        notify_text = (
            f"💸 <b>Devolución de Efectivo</b>\n"
            f"👤 <b>{user.name}</b> devolvió <b>{amount:.2f} €</b> de sobrante.\n"
            f"📝 Motivo: {note or 'Sobrante de caja'}\n"
            f"📊 Nuevo estado del trabajador: {bal.status_label}"
        )
        await notify_admins(bot, repo, admin_telegram_ids, notify_text)

    # ── Puse de mi Bolsillo Flow ──────────────────────────────────────────

    @router.message(StateFilter("*"), Command("bolsillo"))
    @router.message(StateFilter("*"), F.text == BTN_BOLSILLO)
    async def handle_own_pocket_start(message: Message, state: FSMContext | None = None) -> None:
        """Start out-of-pocket cash declaration flow."""
        await _clear_state(state)
        user = repo.get_user_by_telegram_id(message.from_user.id)
        if user is None:
            await message.answer("Primero escribí /start para registrarte.")
            return

        if state is not None:
            await state.set_state(OwnPocketStates.waiting_for_amount)
        await message.answer(
            "👛 <b>Puse de mi Bolsillo</b>\n\n"
            "¿Cuánto dinero propio pusiste o disponés hoy para gastos?\n"
            "<i>(Tocá un importe o escribí el monto en euros)</i>\n\n"
            "💡 <i>Nota: El importe a reembolsarte por la empresa se calcula con las facturas reales (tickets) que subas al bot.</i>",
            reply_markup=get_amount_inline_keyboard("pocket_amt", amounts=(20, 50, 100)),
        )

    @router.callback_query(OwnPocketStates.waiting_for_amount, F.data.startswith("pocket_amt:"))
    async def handle_own_pocket_card(callback: CallbackQuery, state: FSMContext) -> None:
        amount = parse_euro_amount(callback.data.split(":", 1)[1])
        if amount is None:
            await callback.answer("Importe no válido.", show_alert=True)
            return
        try:
            await callback.message.edit_text(f"👛 Fondo propio: <b>{amount:.2f} €</b>")
        except Exception:
            pass
        await _advance_amount_to_note(
            callback.message,
            state,
            OwnPocketStates.waiting_for_note,
            amount,
            f"Importe: <b>{amount:.2f} €</b>\n\nIngresá un concepto o nota breve (ej: <code>Fondo propio para gasolina</code>) o tocá el botón de abajo:",
        )
        await callback.answer()

    @router.message(OwnPocketStates.waiting_for_amount)
    async def handle_own_pocket_amount(message: Message, state: FSMContext) -> None:
        amount = parse_euro_amount(message.text)
        if amount is None:
            await message.answer(
                "⚠️ Importe no válido. Por favor ingresá un número en euros (ej: 50):",
                reply_markup=get_cancel_keyboard(),
            )
            return
        await _advance_amount_to_note(
            message,
            state,
            OwnPocketStates.waiting_for_note,
            amount,
            f"Importe: <b>{amount:.2f} €</b>\n\nIngresá un concepto o nota breve (ej: <code>Fondo propio para gasolina</code>) o tocá el botón de abajo:",
        )

    @router.message(OwnPocketStates.waiting_for_note)
    async def handle_own_pocket_note(message: Message, state: FSMContext) -> None:
        data = await state.get_data()
        amount = data["amount"]
        note = sanitize_note(message.text)

        user = repo.get_user_by_telegram_id(message.from_user.id)
        if user is None:
            await state.clear()
            return

        repo.create_cash_movement(
            user_id=user.id,
            created_by_id=user.id,
            movement_type="APORTE_PROPIO",
            amount=amount,
            note=note,
        )

        sync_pending_tickets(excel_path, repo)
        await state.clear()

        await message.answer(
            f"✅ <b>Fondo de bolsillo registrado</b>\n\n"
            f"👛 Declaraste: <b>{amount:.2f} €</b> de tu dinero.\n"
            f"📝 Nota: {note or 'Fondo propio'}\n\n"
            "📸 Recordá subir la foto de cada ticket que pagues para que se compute a tu favor.",
            reply_markup=get_user_menu(user, admin_telegram_ids),
        )

        notify_text = (
            f"👛 <b>Aviso de Bolsillo Propio</b>\n"
            f"👤 <b>{user.name}</b> avisó que puso <b>{amount:.2f} €</b> de su dinero para gastos.\n"
            f"📝 Motivo: {note or 'Sin nota'}"
        )
        await notify_admins(bot, repo, admin_telegram_ids, notify_text)

    # ── Ticket Management: Edit Total, Delete, and Admin Ticket List ──────

    @router.callback_query(F.data == "admin_tickets_menu")
    async def handle_admin_tickets_menu(callback: CallbackQuery) -> None:
        """List recent tickets across the company so admin can inspect/edit/void them."""
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        recent = repo.get_recent_tickets(user_id=None, limit=8)
        if not recent:
            await callback.answer("Aún no hay tickets registrados.", show_alert=True)
            return
        await callback.message.answer(
            "🧾 <b>Últimos Tickets Registrados</b>\n\nSeleccioná un ticket para corregir su importe o anularlo:",
            reply_markup=get_manage_tickets_keyboard(recent),
        )
        await callback.answer()

    @router.callback_query(F.data.startswith("tkt_pick:"))
    async def handle_ticket_pick(callback: CallbackQuery) -> None:
        """Show details and action buttons for a specific ticket (Admin only)."""
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        ticket_id = int(callback.data.split(":")[1])
        ticket = repo.get_ticket_by_id(ticket_id)
        if ticket is None:
            await callback.answer("Ticket no encontrado.", show_alert=True)
            return
        owner = repo.get_user_by_id(ticket.user_id)
        owner_name = owner.name if owner else "Desconocido"
        await callback.message.edit_text(
            f"🧾 <b>Ticket #{ticket.id}</b>\n"
            f"👤 Trabajador: <b>{owner_name}</b>\n"
            f"🏢 {ticket.empresa}\n"
            f"📝 {ticket.descripcion}\n"
            f"💰 <b>{ticket.total:.2f} €</b>\n"
            f"📅 {ticket.fecha}",
            reply_markup=get_ticket_actions_inline_keyboard(ticket.id),
        )
        await callback.answer()

    @router.callback_query(F.data.startswith("tkt_edit:"))
    async def handle_ticket_edit_start(callback: CallbackQuery, state: FSMContext) -> None:
        """Prompt user/admin to enter corrected Euro total for a ticket."""
        actor = repo.get_user_by_telegram_id(callback.from_user.id)
        if actor is None:
            await callback.answer("Usuario no registrado.", show_alert=True)
            return
        ticket_id = int(callback.data.split(":")[1])
        ticket = repo.get_ticket_by_id(ticket_id)
        if ticket is None:
            await callback.answer("El ticket ya no existe.", show_alert=True)
            return
        if ticket.user_id != actor.id and not is_admin(actor, admin_telegram_ids):
            await callback.answer("⛔ No tenés permiso para modificar este ticket.", show_alert=True)
            return

        await state.update_data(edit_ticket_id=ticket.id)
        await state.set_state(EditTicketTotalStates.waiting_for_total)
        await callback.message.answer(
            f"✏️ <b>Corregir Importe — {ticket.empresa}</b>\n"
            f"Importe actual: <b>{ticket.total:.2f} €</b>\n\n"
            "Escribí el importe total correcto en euros (ej: <code>14.80</code>):",
            reply_markup=get_cancel_keyboard(),
        )
        await callback.answer()

    @router.message(EditTicketTotalStates.waiting_for_total)
    async def handle_ticket_edit_total(message: Message, state: FSMContext) -> None:
        actor = repo.get_user_by_telegram_id(message.from_user.id)
        if actor is None:
            await state.clear()
            return
        new_total = parse_euro_amount(message.text)
        if new_total is None:
            await message.answer(
                "⚠️ Importe no válido. Escribí un importe positivo en euros (ej: 14.80):",
                reply_markup=get_cancel_keyboard(),
            )
            return

        data = await state.get_data()
        ticket_id = data.get("edit_ticket_id")
        ticket = repo.get_ticket_by_id(ticket_id) if ticket_id is not None else None
        if ticket is None or (ticket.user_id != actor.id and not is_admin(actor, admin_telegram_ids)):
            await state.clear()
            await message.answer("⛔ Ticket no encontrado o sin permisos.", reply_markup=get_user_menu(actor, admin_telegram_ids))
            return

        old_total = ticket.total
        repo.update_ticket(ticket.id, total=new_total, synced_to_excel=False)
        rebuild_excel_from_db(excel_path, repo)
        bal = repo.get_user_balance(ticket.user_id)
        await state.clear()

        await message.answer(
            f"✅ <b>Ticket #{ticket.id} actualizado</b>\n\n"
            f"🏢 {ticket.empresa}\n"
            f"💰 {old_total:.2f} € ➡️ <b>{new_total:.2f} €</b>\n"
            f"📊 Saldo actualizado: <b>{bal.status_label}</b>",
            reply_markup=get_user_menu(actor, admin_telegram_ids),
        )

    @router.callback_query(F.data.startswith("tkt_del:"))
    async def handle_ticket_delete_prompt(callback: CallbackQuery) -> None:
        """Ask confirmation before deleting a ticket."""
        actor = repo.get_user_by_telegram_id(callback.from_user.id)
        if actor is None:
            await callback.answer("Usuario no registrado.", show_alert=True)
            return
        ticket_id = int(callback.data.split(":")[1])
        ticket = repo.get_ticket_by_id(ticket_id)
        if ticket is None:
            await callback.answer("El ticket ya fue eliminado.", show_alert=True)
            return
        if ticket.user_id != actor.id and not is_admin(actor, admin_telegram_ids):
            await callback.answer("⛔ No tenés permiso para anular este ticket.", show_alert=True)
            return

        await callback.message.answer(
            f"⚠️ ¿Confirmás anular este ticket?\n"
            f"🏢 <b>{ticket.empresa}</b> — 💰 <b>{ticket.total:.2f} €</b> ({ticket.fecha})",
            reply_markup=get_confirm_delete_ticket_keyboard(ticket.id),
        )
        await callback.answer()

    @router.callback_query(F.data.startswith("tkt_del_confirm:"))
    async def handle_ticket_delete_confirm(callback: CallbackQuery) -> None:
        """Delete ticket from SQLite, delete image file, and rebuild Excel."""
        actor = repo.get_user_by_telegram_id(callback.from_user.id)
        if actor is None:
            await callback.answer("Usuario no registrado.", show_alert=True)
            return
        ticket_id = int(callback.data.split(":")[1])
        ticket = repo.get_ticket_by_id(ticket_id)
        if ticket is None:
            await callback.answer("El ticket ya no existe.", show_alert=True)
            return
        if ticket.user_id != actor.id and not is_admin(actor, admin_telegram_ids):
            await callback.answer("⛔ No tenés permiso para anular este ticket.", show_alert=True)
            return

        deleted = repo.delete_ticket(ticket.id)
        if deleted is None:
            await callback.answer("No se pudo anular el ticket.", show_alert=True)
            return

        if deleted.image_path:
            Path(deleted.image_path).unlink(missing_ok=True)
        rebuild_excel_from_db(excel_path, repo)
        bal = repo.get_user_balance(deleted.user_id)

        await callback.message.edit_text(
            f"🗑️ <b>Ticket #{deleted.id} anulado</b>\n\n"
            f"🏢 {deleted.empresa} — {deleted.total:.2f} €\n"
            f"📊 Saldo recalculado: <b>{bal.status_label}</b>"
        )
        await callback.answer("Ticket anulado")

    # ── Worker Management (Admin Only) ────────────────────────────────────

    @router.callback_query(F.data == "admin_workers_menu")
    async def handle_admin_workers_menu(callback: CallbackQuery) -> None:
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        workers = repo.get_all_users()
        if not workers:
            await callback.answer("No hay trabajadores registrados.", show_alert=True)
            return
        await callback.message.answer(
            "👥 <b>Gestión de Trabajadores</b>\n\nElegí un trabajador:",
            reply_markup=get_workers_inline_keyboard(workers, callback_prefix="mgr_worker", include_create_dummy=False),
        )
        await callback.answer()

    @router.callback_query(F.data.startswith("mgr_worker:"))
    async def handle_manage_worker_chosen(callback: CallbackQuery) -> None:
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        worker_id = int(callback.data.split(":")[1])
        worker = repo.get_user_by_id(worker_id)
        if worker is None:
            await callback.answer("Trabajador no encontrado.", show_alert=True)
            return
        is_env_super = worker.telegram_id in admin_telegram_ids
        role_label = "Admin 👑" if worker.role == "admin" else "Trabajador 👤"
        await callback.message.edit_text(
            f"👤 <b>{worker.name}</b>\n"
            f"Rol: <b>{role_label}</b>\n"
            f"Telegram ID: <code>{worker.telegram_id}</code>",
            reply_markup=get_worker_actions_keyboard(worker, is_env_superadmin=is_env_super),
        )
        await callback.answer()

    @router.callback_query(F.data.startswith("wrk_rename:"))
    async def handle_worker_rename_start(callback: CallbackQuery, state: FSMContext) -> None:
        admin_user = await _require_admin_cb(callback, state)
        if admin_user is None:
            return
        worker_id = int(callback.data.split(":")[1])
        worker = repo.get_user_by_id(worker_id)
        if worker is None:
            await callback.answer("Trabajador no encontrado.", show_alert=True)
            return
        await state.update_data(rename_worker_id=worker.id, old_worker_name=worker.name)
        await state.set_state(RenameWorkerStates.waiting_for_name)
        await callback.message.answer(
            f"✏️ Escribí el nuevo nombre para <b>{worker.name}</b>:",
            reply_markup=get_cancel_keyboard(),
        )
        await callback.answer()

    @router.message(RenameWorkerStates.waiting_for_name)
    async def handle_worker_rename_save(message: Message, state: FSMContext) -> None:
        admin_user = await _require_admin_msg(message, state)
        if admin_user is None:
            return
        new_name = message.text.strip()
        if len(new_name) < 2 or len(new_name) > 100:
            await message.answer("El nombre debe tener entre 2 y 100 caracteres:")
            return
        data = await state.get_data()
        worker_id = data["rename_worker_id"]
        old_name = data["old_worker_name"]
        repo.update_user_name(worker_id, new_name)
        rebuild_excel_from_db(excel_path, repo)
        await state.clear()
        await message.answer(
            f"✅ Trabajador renombrado: <b>{old_name}</b> ➡️ <b>{new_name}</b>\n"
            "<i>El archivo Excel se ha actualizado con el nuevo nombre.</i>",
            reply_markup=get_user_menu(admin_user, admin_telegram_ids),
        )

    @router.callback_query(F.data.startswith("wrk_del_dummy:"))
    async def handle_worker_delete_dummy(callback: CallbackQuery) -> None:
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        worker_id = int(callback.data.split(":")[1])
        deleted = repo.delete_dummy_worker(worker_id)
        if not deleted:
            await callback.answer(
                "⚠️ Solo se pueden borrar trabajadores de prueba sin tickets ni movimientos.",
                show_alert=True,
            )
            return
        await callback.message.edit_text("🗑️ Trabajador de prueba eliminado correctamente.")
        await callback.answer("Eliminado")

    @router.callback_query(F.data.startswith("wrk_role:"))
    async def handle_worker_toggle_role(callback: CallbackQuery) -> None:
        admin_user = await _require_admin_cb(callback)
        if admin_user is None:
            return
        parts = callback.data.split(":")
        worker_id, new_role = int(parts[1]), parts[2]
        if new_role not in ("admin", "trabajador"):
            await callback.answer("Rol inválido.", show_alert=True)
            return
        worker = repo.get_user_by_id(worker_id)
        if worker is None:
            await callback.answer("Trabajador no encontrado.", show_alert=True)
            return
        if worker.telegram_id in admin_telegram_ids and new_role != "admin":
            await callback.answer("No podés quitar el rol admin a un administrador definido en .env.", show_alert=True)
            return

        promoted = new_role == "admin"
        repo.set_user_role(worker.id, new_role, admin_source="bot" if promoted else "")
        rebuild_excel_from_db(excel_path, repo)

        if worker.telegram_id > 0:
            await sync_user_chat_commands(bot, worker.telegram_id, is_admin=promoted)
            notify_msg = (
                "👑 <b>¡Has sido ascendido a Administrador!</b>\n\n"
                "Ya tenés activado <b>👑 Panel Admin</b>, asignación de caja y descarga de Excel en tu menú inferior."
                if promoted
                else "ℹ️ <b>Tu rol ha cambiado a Trabajador 👤</b>\n\n"
                "Tu menú inferior se actualizó a las opciones de trabajador."
            )
            try:
                await bot.send_message(
                    chat_id=worker.telegram_id,
                    text=notify_msg,
                    reply_markup=get_main_keyboard(is_admin=promoted),
                )
            except Exception as e:
                logger.warning(f"Could not push keyboard update to {worker.telegram_id}: {e}")

        await callback.message.edit_text(
            f"✅ Rol de <b>{worker.name}</b> cambiado a <b>{new_role.upper()}</b>.\n"
            "<i>Su menú y Excel se actualizaron automáticamente.</i>"
        )
        await callback.answer("Rol actualizado")

    # ── Duplicate Receipt Confirmation Callbacks ──────────────────────────

    @router.callback_query(F.data.startswith("dup_drop:"))
    async def handle_duplicate_drop(callback: CallbackQuery) -> None:
        """Discard a duplicate photo and delete temporary file."""
        pending_id = callback.data.split(":", 1)[1]
        pending = pending_duplicates.get(pending_id)
        if pending is None:
            await callback.answer("Esta confirmación ya caducó.", show_alert=True)
            return
        if pending.telegram_id != callback.from_user.id:
            await callback.answer("⛔ Esta confirmación pertenece a otro usuario.", show_alert=True)
            return

        pending_duplicates.pop(pending_id, None)
        pending.image_path.unlink(missing_ok=True)
        await callback.message.edit_text("🗑️ <b>Foto repetida descartada.</b> Tu saldo no se ha modificado.")
        await callback.answer("Descartada")

    @router.callback_query(F.data.startswith("dup_keep:"))
    async def handle_duplicate_keep(callback: CallbackQuery) -> None:
        """User confirmed receipt is distinct — complete OCR if needed and save."""
        pending_id = callback.data.split(":", 1)[1]
        pending = pending_duplicates.get(pending_id)
        if pending is None:
            await callback.answer("Esta confirmación ya caducó.", show_alert=True)
            return
        if pending.telegram_id != callback.from_user.id:
            await callback.answer("⛔ Esta confirmación pertenece a otro usuario.", show_alert=True)
            return

        user = repo.get_user_by_id(pending.user_id)
        if user is None:
            pending_duplicates.pop(pending_id, None)
            pending.image_path.unlink(missing_ok=True)
            await callback.answer("Usuario no encontrado.", show_alert=True)
            return

        pending_duplicates.pop(pending_id, None)
        await callback.answer("Procesando...")

        try:
            extracted = pending.extracted
            if extracted is None:
                raw_bytes = pending.image_path.read_bytes()
                resized = resize_if_large(raw_bytes, max_width=1024)
                extracted = await extractor.extract_from_image(resized)
                if not is_valid_ticket(extracted):
                    pending.image_path.unlink(missing_ok=True)
                    await callback.message.edit_text(
                        "❌ <b>No pude detectar un ticket válido</b> (o el importe es 0.00 €).\n"
                        "<i>(Esta imagen no se ha guardado)</i>."
                    )
                    return

            try:
                await callback.message.delete()
            except Exception:
                pass
            await _persist_and_confirm_ticket(
                callback.message,
                user,
                pending.image_path,
                extracted,
                pending.sha256,
                pending.dhash,
            )
        except OllamaUnavailableError as e:
            pending.image_path.unlink(missing_ok=True)
            logger.error(f"Ollama unavailable on dup_keep: {e}")
            await callback.message.answer(
                "⚠️ <b>El motor de IA local no está disponible en este momento.</b>",
                reply_markup=get_user_menu(user, admin_telegram_ids),
            )
        except Exception as e:
            pending.image_path.unlink(missing_ok=True)
            logger.error(f"Error processing duplicate keep: {type(e).__name__}: {e}")
            await callback.message.answer(
                "⚠️ No pude procesar ese ticket. Mandalo de nuevo.",
                reply_markup=get_user_menu(user, admin_telegram_ids),
            )

    # ── Ticket Photo Processing ───────────────────────────────────────────

    @router.message(StateFilter("*"), F.photo)
    async def handle_photo(message: Message) -> None:
        """Handle photo messages — detect duplicates, extract OCR, and save ticket."""
        telegram_id = message.from_user.id
        user = repo.get_user_by_telegram_id(telegram_id)

        if user is None:
            await message.answer("Primero escribí /start para registrarte.")
            return

        _prune_pending_duplicates()

        try:
            await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        except Exception:
            pass

        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        image_bytes = await bot.download_file(file.file_path)

        if image_bytes is None:
            return

        raw_bytes = image_bytes.read()
        image_name = f".tmp_{uuid.uuid4().hex}.jpg"
        image_path = images_dir / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(image_path.parent, 0o755)
        except OSError:
            pass
        image_path.write_bytes(raw_bytes)
        try:
            os.chmod(image_path, 0o644)
        except OSError:
            pass

        try:
            resized = resize_if_large(raw_bytes, max_width=1024)
            logger.info(f"Image ready: {len(raw_bytes)} -> {len(resized)} bytes (resized)")

            sha256 = compute_sha256(raw_bytes)
            dhash = compute_dhash(resized)

            # Layer 1: check exact / visual duplicate BEFORE calling Gemini API
            img_dup = repo.find_duplicate_by_image_hash(sha256, dhash, max_hamming=4, days_window=90)
            if img_dup is not None:
                orig_ticket, orig_owner, _kind = img_dup
                pending_id = uuid.uuid4().hex[:10]
                pending_duplicates[pending_id] = _PendingDuplicate(
                    telegram_id=telegram_id,
                    user_id=user.id,
                    image_path=image_path,
                    sha256=sha256,
                    dhash=dhash,
                    created_ts=time.monotonic(),
                    extracted=None,
                )
                await message.answer(
                    "⚠️ <b>¡Ojo! Esta foto parece REPETIDA</b>\n\n"
                    "Ya existe una foto igual/similar registrada:\n"
                    f"👤 Subida por: <b>{orig_owner}</b>\n"
                    f"🏢 {orig_ticket.empresa} — 💰 <b>{orig_ticket.total:.2f} €</b> ({orig_ticket.fecha})\n\n"
                    "¿Querés descartarla o procesarla igualmente?",
                    reply_markup=get_duplicate_decision_keyboard(pending_id),
                )
                return

            extracted = await extractor.extract_from_image(resized)
            logger.info(
                f"Extraction: {extracted.empresa} - {extracted.total:.2f} € "
                f"(es_ticket={extracted.es_ticket})"
            )

            if not is_valid_ticket(extracted):
                image_path.unlink(missing_ok=True)
                logger.info(
                    f"Rejected non-ticket image: total={extracted.total:.2f}, "
                    f"es_ticket={extracted.es_ticket}"
                )
                await message.answer(
                    "❌ <b>No pude detectar un ticket válido</b> (o el importe es 0.00 €).\n\n"
                    "📸 Si es un ticket de gasto real, por favor enviá una foto más nítida donde se vea el total claramente.\n"
                    "<i>(Esta imagen no se ha guardado en tu cuenta ni altera tu saldo)</i>.",
                    reply_markup=get_user_menu(user, admin_telegram_ids),
                )
                return

            # Layer 2: check same receipt photographed twice (indexed fecha + total + empresa)
            data_dup = repo.find_duplicate_by_data(
                extracted.empresa,
                extracted.total,
                extracted.fecha,
                days_window=90,
            )
            if data_dup is not None:
                orig_ticket, orig_owner = data_dup
                pending_id = uuid.uuid4().hex[:10]
                pending_duplicates[pending_id] = _PendingDuplicate(
                    telegram_id=telegram_id,
                    user_id=user.id,
                    image_path=image_path,
                    sha256=sha256,
                    dhash=dhash,
                    created_ts=time.monotonic(),
                    extracted=extracted,
                )
                await message.answer(
                    "⚠️ <b>¡Posible Ticket Duplicado!</b>\n\n"
                    "Ya existe otro ticket con la misma fecha, comercio e importe:\n"
                    f"👤 Subido por: <b>{orig_owner}</b>\n"
                    f"🏢 {orig_ticket.empresa} — 💰 <b>{orig_ticket.total:.2f} €</b> ({orig_ticket.fecha})\n\n"
                    "¿Querés descartarlo o guardarlo igualmente?",
                    reply_markup=get_duplicate_decision_keyboard(pending_id),
                )
                return

            await _persist_and_confirm_ticket(message, user, image_path, extracted, sha256, dhash)

        except OllamaUnavailableError as e:
            image_path.unlink(missing_ok=True)
            logger.error(f"Ollama unavailable: {e}")
            await message.answer(
                "⚠️ <b>El motor de IA local (Ollama) no está iniciado.</b>\n\n"
                "💻 Por favor, abrí Ollama en el equipo y volvé a enviar la foto.",
                reply_markup=get_user_menu(user, admin_telegram_ids),
            )

        except Exception as e:
            image_path.unlink(missing_ok=True)
            logger.error(f"Error processing ticket: {type(e).__name__}: {e}")
            await message.answer(
                "⚠️ No pude procesar ese ticket. "
                "Asegurate de que la foto sea clara y mandala de nuevo.",
                reply_markup=get_user_menu(user, admin_telegram_ids),
            )

    # ── Plain text / User Registration (catch-all outside menu/FSM states) ──

    @router.message(F.text & ~F.text.startswith("/") & ~F.text.in_(ALL_MENU_BUTTONS))
    async def handle_text(message: Message) -> None:
        """Handle plain text — if user is new, register their name."""
        telegram_id = message.from_user.id
        existing = repo.get_user_by_telegram_id(telegram_id)

        if existing is not None:
            await message.answer(
                f"📸 {existing.name}, mandame la foto de tu ticket para procesarlo, o usá los botones de abajo:",
                reply_markup=get_user_menu(existing, admin_telegram_ids),
            )
            return

        name = message.text.strip()
        if len(name) < 2 or len(name) > 100:
            await message.answer(
                "El nombre debe tener entre 2 y 100 caracteres. "
                "Escribí tu nombre completo."
            )
            return

        is_env_admin = telegram_id in admin_telegram_ids
        role = "admin" if is_env_admin else "trabajador"
        new_user = repo.create_user(
            telegram_id=telegram_id,
            name=name,
            role=role,
            admin_source="env" if is_env_admin else "",
        )
        await sync_user_chat_commands(bot, telegram_id, is_admin=is_env_admin)

        role_msg = (
            "👑 Has sido registrado como <b>Administrador</b>."
            if role == "admin"
            else "👤 Has sido registrado como <b>Trabajador</b>."
        )
        await message.answer(
            f"✅ Registrado como {name}.\n"
            f"{role_msg}\n\n"
            "📸 Mandame fotos de tus tickets o usá las opciones de abajo:",
            reply_markup=get_user_menu(new_user, admin_telegram_ids),
        )

    return router
