"""Telegram bot message handlers with RBAC and accounting workflows."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from aiogram import Bot, Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from db.models import User
from db.repository import Repository
from ocr.preprocessor import resize_if_large
from ocr.extractor import VisionExtractor, is_suspicious_total, is_valid_ticket
from excel.writer import sync_pending_tickets
from bot.keyboards import (
    BTN_RESUMEN,
    BTN_RESUMEN_LEGACY,
    BTN_PANEL_JEFA,
    BTN_ASIGNAR_DINERO,
    BTN_DEVOLVER,
    BTN_BOLSILLO,
    BTN_SINCRONIZAR,
    BTN_AYUDA,
    BTN_CANCELAR,
    BTN_SIN_NOTA,
    get_main_keyboard,
    get_cancel_keyboard,
    get_note_keyboard,
    get_sync_inline_keyboard,
    get_workers_inline_keyboard,
)

logger = logging.getLogger(__name__)


# ── FSM States for interactive flows ───────────────────────────────────────

class AssignMoneyStates(StatesGroup):
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


# ── Helpers ───────────────────────────────────────────────────────────────

def is_admin(user: User | None, admin_telegram_ids: tuple[int, ...] = ()) -> bool:
    """Check if user has admin privileges."""
    if user is None:
        return False
    if user.role == "admin":
        return True
    if user.telegram_id in admin_telegram_ids:
        return True
    return False


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
    """Safely parse user entered money amount (e.g. '50', '50.50', '50,50 €')."""
    cleaned = text.replace("€", "").replace("eur", "").replace("EUR", "").strip()
    cleaned = cleaned.replace(",", ".")
    try:
        val = float(cleaned)
        return round(val, 2) if val > 0 else None
    except ValueError:
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

    # ── Start & Cancel ────────────────────────────────────────────────────

    @router.message(Command("start"))
    async def handle_start(message: Message, state: FSMContext | None = None) -> None:
        """Handle /start command — register user if new, or show welcome back."""
        if state is not None:
            await state.clear()
        telegram_id = message.from_user.id
        existing = repo.get_user_by_telegram_id(telegram_id)

        if existing:
            # Auto-promote to admin if listed in settings
            if telegram_id in admin_telegram_ids and existing.role != "admin":
                repo.set_user_role(existing.id, "admin")
                existing.role = "admin"

            role_desc = "Jefa / Admin 👑" if is_admin(existing, admin_telegram_ids) else "Trabajador 👤"
            await message.answer(
                f"👋 ¡Hola de nuevo, {existing.name}!\n"
                f"Rol actual: <b>{role_desc}</b>\n\n"
                "📸 Mandame fotos de tus tickets o usá los botones de abajo:",
                reply_markup=get_user_menu(existing, admin_telegram_ids),
            )
            return

        # Ask for name
        await message.answer(
            "👋 ¡Hola! Soy el bot de tickets y control de caja.\n\n"
            "¿Cómo te llamás? Escribime tu nombre completo (ej: Juan Pérez)."
        )

    @router.message(Command("cancelar"))
    @router.message(F.text == BTN_CANCELAR)
    async def handle_cancel_text(message: Message, state: FSMContext | None = None) -> None:
        """Cancel any active FSM dialog via reply button."""
        if state is not None:
            await state.clear()
        user = repo.get_user_by_telegram_id(message.from_user.id)
        await message.answer(
            "❌ Acción cancelada. Volvemos al menú principal.",
            reply_markup=get_user_menu(user, admin_telegram_ids),
        )

    @router.callback_query(F.data == "cancel_action")
    async def handle_cancel_callback(callback: CallbackQuery, state: FSMContext | None = None) -> None:
        """Cancel any active FSM dialog via inline button."""
        if state is not None:
            await state.clear()
        try:
            if callback.message:
                await callback.message.delete()
        except Exception:
            pass
        user = repo.get_user_by_telegram_id(callback.from_user.id)
        await callback.message.answer(
            "❌ Acción cancelada.",
            reply_markup=get_user_menu(user, admin_telegram_ids),
        )
        await callback.answer()

    # ── Help ──────────────────────────────────────────────────────────────

    @router.message(Command("ayuda"))
    @router.message(F.text == BTN_AYUDA)
    async def handle_help(message: Message) -> None:
        """Handle /ayuda command or Ayuda button — show role-tailored guide."""
        user = repo.get_user_by_telegram_id(message.from_user.id)
        admin_mode = is_admin(user, admin_telegram_ids)

        if admin_mode:
            text = (
                "👑 <b>Panel de Control de la Jefa</b>\n\n"
                "• <b>👑 Panel Jefa (Saldos):</b> Ver el balance de todos los trabajadores (quién está a paz y salvo, quién debe dinero y a quién se le debe).\n"
                "• <b>💵 Asignar Dinero:</b> Entregar dinero en efectivo a un trabajador.\n"
                "• <b>📊 Mi Cuenta / Saldo:</b> Ver tu balance individual.\n"
                "• <b>💸 Devolver Sobrante:</b> Registrar devolución de efectivo.\n"
                "• <b>🔄 Sincronizar Excel:</b> Actualizar tickets y saldos en tickets.xlsx.\n"
                "• 📸 Podés mandarme fotos de tickets en cualquier momento."
            )
        else:
            text = (
                "👤 <b>Guía del Trabajador</b>\n\n"
                "• 📸 <b>Enviar Ticket:</b> Mandá una foto clara de cualquier ticket/factura y lo leo al instante.\n"
                "• <b>📊 Mi Cuenta / Saldo:</b> Consultá cuánto dinero te asignó la jefa, cuánto gastaste y tu saldo.\n"
                "• <b>💸 Devolver Sobrante:</b> Registrá cuando devuelvas dinero en efectivo a la jefa.\n"
                "• <b>👛 Puse de mi Bolsillo:</b> Dejá constancia si pusiste dinero propio hoy.\n"
                "• <b>🔄 Sincronizar Excel:</b> Pasar datos pendientes al archivo Excel."
            )

        await message.answer(text, reply_markup=get_user_menu(user, admin_telegram_ids))

    # ── Synchronize Excel ─────────────────────────────────────────────────

    @router.message(Command("sincronizar", "sync"))
    @router.message(F.text == BTN_SINCRONIZAR)
    async def handle_sync(message: Message) -> None:
        """Handle /sincronizar command or button — manually trigger Excel sync."""
        user = repo.get_user_by_telegram_id(message.from_user.id)
        if user is None:
            await message.answer("Primero escribí /start para registrarte.")
            return

        synced_count, sync_ok = sync_pending_tickets(excel_path, repo)

        if not sync_ok:
            await message.answer(
                "⚠️ El archivo Excel sigue abierto o bloqueado en el equipo.\n\n"
                "Por favor cerralo y volvé a tocar 🔄 Sincronizar Excel.",
                reply_markup=get_sync_inline_keyboard(),
            )
            return

        if synced_count > 0:
            await message.answer(
                f"✅ ¡Sincronización exitosa!\n"
                f"Se volcaron {synced_count} registro(s) pendiente(s) al archivo Excel.",
                reply_markup=get_user_menu(user, admin_telegram_ids),
            )
        else:
            await message.answer(
                "✅ El archivo Excel ya está completamente al día.",
                reply_markup=get_user_menu(user, admin_telegram_ids),
            )

    @router.callback_query(F.data == "do_sync")
    async def handle_callback_sync(callback: CallbackQuery) -> None:
        """Handle inline button press for immediate sync."""
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

    # ── Mi Cuenta / Saldo ─────────────────────────────────────────────────

    @router.message(Command("resumen", "saldo"))
    @router.message(F.text == BTN_RESUMEN)
    @router.message(F.text == BTN_RESUMEN_LEGACY)
    async def handle_summary(message: Message) -> None:
        """Show user's comprehensive accounting balance and last tickets."""
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
                f"🟡 <i>Tenés <b>{bal.net_balance:.2f} €</b> de la Jefa pendientes de justificar con tickets o devolver.</i>"
            )
        else:
            lines.append(
                f"🔴 <i>La Jefa te debe reembolsar <b>{abs(bal.net_balance):.2f} €</b> de tickets pagados con tu dinero.</i>"
            )

        if tickets:
            lines.append("\n<b>Últimos tickets subidos:</b>")
            for t in tickets[:5]:
                lines.append(f"• {t.fecha} — {t.empresa}: {t.total:.2f} €")

        await message.answer("\n".join(lines), reply_markup=get_user_menu(user, admin_telegram_ids))

    # ── Panel Jefa (Admin Only) ───────────────────────────────────────────

    @router.message(Command("panel"))
    @router.message(F.text == BTN_PANEL_JEFA)
    async def handle_boss_panel(message: Message) -> None:
        """Show boss control panel with balances for all workers."""
        user = repo.get_user_by_telegram_id(message.from_user.id)
        if not is_admin(user, admin_telegram_ids):
            await message.answer("⛔ Esta opción es exclusiva para la Jefa / Administradores.")
            return

        balances = repo.get_all_balances()
        if not balances:
            await message.answer("Aún no hay trabajadores registrados en el sistema.")
            return

        month_label = balances[0].month_label if balances else ""
        lines = [
            f"👑 <b>Panel de Control de la Jefa ({month_label})</b>",
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
            "\n💡 <i>Toca '💵 Asignar Dinero' para entregar fondos o consulta el Excel para el detalle completo.</i>",
        ])

        await message.answer("\n".join(lines), reply_markup=get_user_menu(user, admin_telegram_ids))

    # ── Asignar Dinero (Admin Only Flow) ──────────────────────────────────

    @router.message(Command("asignar"))
    @router.message(F.text == BTN_ASIGNAR_DINERO)
    async def handle_assign_money_start(message: Message) -> None:
        """Start assign money flow by picking a worker."""
        user = repo.get_user_by_telegram_id(message.from_user.id)
        if not is_admin(user, admin_telegram_ids):
            await message.answer("⛔ Esta opción es exclusiva para la Jefa / Administradores.")
            return

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
        worker_id = int(callback.data.split(":")[1])
        worker = repo.get_user_by_id(worker_id)
        if worker is None:
            await callback.answer("Trabajador no encontrado.", show_alert=True)
            return

        await state.update_data(target_worker_id=worker.id, target_worker_name=worker.name)
        await state.set_state(AssignMoneyStates.waiting_for_amount)

        await callback.message.edit_text(
            f"💵 Asignar dinero a <b>{worker.name}</b>\n\n"
            "¿Cuánto dinero en efectivo le vas a entregar en euros?\n"
            "<i>(Escribí el importe, ej: 50 o 75.50)</i>"
        )
        await callback.message.answer(
            "Podés pulsar el botón de abajo para cancelar en cualquier momento:",
            reply_markup=get_cancel_keyboard(),
        )
        await callback.answer()

    @router.callback_query(F.data == "asig_create_dummy")
    async def handle_create_dummy_start(callback: CallbackQuery, state: FSMContext) -> None:
        """Allow admin to create a test worker for instant multi-user simulation without a 2nd phone."""
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
        name = message.text.strip()
        if len(name) < 2 or len(name) > 100:
            await message.answer("El nombre debe tener entre 2 y 100 caracteres. Escribí otro nombre:")
            return

        dummy = repo.create_dummy_worker(name)
        await state.clear()
        user = repo.get_user_by_telegram_id(message.from_user.id)

        await message.answer(
            f"✅ Se creó el trabajador de prueba <b>{dummy.name}</b>.\n"
            "Ahora ya aparece en el panel y podés asignarle dinero para probar.",
            reply_markup=get_user_menu(user, admin_telegram_ids),
        )

    @router.message(AssignMoneyStates.waiting_for_amount)
    async def handle_assign_amount(message: Message, state: FSMContext) -> None:
        amount = parse_euro_amount(message.text)
        if amount is None:
            await message.answer(
                "⚠️ Importe no válido. Por favor ingresá un número positivo en euros (ej: 50 o 50.50):",
                reply_markup=get_cancel_keyboard(),
            )
            return

        await state.update_data(amount=amount)
        await state.set_state(AssignMoneyStates.waiting_for_note)
        await message.answer(
            f"Importe: <b>{amount:.2f} €</b>\n\n"
            "Ingresá un concepto o motivo (ej: <code>Gasolina semana</code>) o tocá el botón de abajo:",
            reply_markup=get_note_keyboard(),
        )

    @router.message(AssignMoneyStates.waiting_for_note)
    async def handle_assign_note(message: Message, state: FSMContext) -> None:
        data = await state.get_data()
        worker_id = data["target_worker_id"]
        worker_name = data["target_worker_name"]
        amount = data["amount"]

        raw_note = message.text.strip()
        note = "" if raw_note in ("-", ".", BTN_SIN_NOTA, "Sin Nota", "sin nota") else raw_note

        admin_user = repo.get_user_by_telegram_id(message.from_user.id)
        admin_id = admin_user.id if admin_user else worker_id

        # Save movement
        repo.create_cash_movement(
            user_id=worker_id,
            created_by_id=admin_id,
            movement_type="ASIGNACION",
            amount=amount,
            note=note,
        )

        # Sync to Excel
        sync_pending_tickets(excel_path, repo)
        await state.clear()

        await message.answer(
            f"✅ <b>Asignación registrada con éxito</b>\n\n"
            f"👤 Trabajador: <b>{worker_name}</b>\n"
            f"💵 Importe: <b>{amount:.2f} €</b>\n"
            f"📝 Nota: {note or 'Sin nota'}",
            reply_markup=get_user_menu(admin_user, admin_telegram_ids),
        )

        # Cross-notification to worker
        worker = repo.get_user_by_id(worker_id)
        if worker and worker.telegram_id > 0:
            notify_text = (
                f"💵 <b>La Jefa te ha entregado {amount:.2f} € en efectivo</b>\n"
                f"📝 Motivo: {note or 'Fondo de caja'}\n\n"
                "Podés ver tu saldo actualizado pulsando <b>📊 Mi Cuenta / Saldo</b>."
            )
            await notify_worker(bot, worker.telegram_id, notify_text)

    # ── Devolver Sobrante Flow ────────────────────────────────────────────

    @router.message(Command("devolver"))
    @router.message(F.text == BTN_DEVOLVER)
    async def handle_return_money_start(message: Message, state: FSMContext) -> None:
        """Start return leftover cash flow."""
        user = repo.get_user_by_telegram_id(message.from_user.id)
        if user is None:
            await message.answer("Primero escribí /start para registrarte.")
            return

        bal = repo.get_user_balance(user.id)
        await state.set_state(ReturnMoneyStates.waiting_for_amount)

        await message.answer(
            f"💸 <b>Devolver Sobrante a la Jefa</b>\n"
            f"Tu saldo pendiente actual: <b>{bal.net_balance:.2f} €</b>\n\n"
            "¿Cuánto dinero en efectivo vas a devolver en euros?\n"
            "<i>(Escribí el importe, ej: 20 o 25.50)</i>",
            reply_markup=get_cancel_keyboard(),
        )

    @router.message(ReturnMoneyStates.waiting_for_amount)
    async def handle_return_amount(message: Message, state: FSMContext) -> None:
        amount = parse_euro_amount(message.text)
        if amount is None:
            await message.answer(
                "⚠️ Importe no válido. Por favor ingresá un número positivo en euros (ej: 20 o 20.50):",
                reply_markup=get_cancel_keyboard(),
            )
            return

        await state.update_data(amount=amount)
        await state.set_state(ReturnMoneyStates.waiting_for_note)
        await message.answer(
            f"Importe a devolver: <b>{amount:.2f} €</b>\n\n"
            "Ingresá un concepto o nota (ej: <code>Sobrante viaje Madrid</code>) o tocá el botón de abajo:",
            reply_markup=get_note_keyboard(),
        )

    @router.message(ReturnMoneyStates.waiting_for_note)
    async def handle_return_note(message: Message, state: FSMContext) -> None:
        data = await state.get_data()
        amount = data["amount"]
        raw_note = message.text.strip()
        note = "" if raw_note in ("-", ".", BTN_SIN_NOTA, "Sin Nota", "sin nota") else raw_note

        user = repo.get_user_by_telegram_id(message.from_user.id)
        if user is None:
            await state.clear()
            return

        # Direct registration without friction
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

        # Notify admins instantly
        notify_text = (
            f"💸 <b>Devolución de Efectivo</b>\n"
            f"👤 <b>{user.name}</b> devolvió <b>{amount:.2f} €</b> de sobrante.\n"
            f"📝 Motivo: {note or 'Sobrante de caja'}\n"
            f"📊 Nuevo estado del trabajador: {bal.status_label}"
        )
        await notify_admins(bot, repo, admin_telegram_ids, notify_text)

    # ── Puse de mi Bolsillo Flow ──────────────────────────────────────────

    @router.message(Command("bolsillo"))
    @router.message(F.text == BTN_BOLSILLO)
    async def handle_own_pocket_start(message: Message, state: FSMContext) -> None:
        """Start out-of-pocket cash declaration flow."""
        user = repo.get_user_by_telegram_id(message.from_user.id)
        if user is None:
            await message.answer("Primero escribí /start para registrarte.")
            return

        await state.set_state(OwnPocketStates.waiting_for_amount)
        await message.answer(
            "👛 <b>Puse de mi Bolsillo</b>\n\n"
            "¿Cuánto dinero propio pusiste o disponés hoy para gastos?\n"
            "<i>(Escribí el importe, ej: 50 o 50.00)</i>\n\n"
            "💡 <i>Nota: La deuda que la Jefa debe reembolsarte se calcula con las facturas reales (tickets) que subas al bot.</i>",
            reply_markup=get_cancel_keyboard(),
        )

    @router.message(OwnPocketStates.waiting_for_amount)
    async def handle_own_pocket_amount(message: Message, state: FSMContext) -> None:
        amount = parse_euro_amount(message.text)
        if amount is None:
            await message.answer(
                "⚠️ Importe no válido. Por favor ingresá un número en euros (ej: 50):",
                reply_markup=get_cancel_keyboard(),
            )
            return

        await state.update_data(amount=amount)
        await state.set_state(OwnPocketStates.waiting_for_note)
        await message.answer(
            f"Importe: <b>{amount:.2f} €</b>\n\n"
            "Ingresá un concepto o nota breve (ej: <code>Fondo propio para gasolina</code>) o tocá el botón de abajo:",
            reply_markup=get_note_keyboard(),
        )

    @router.message(OwnPocketStates.waiting_for_note)
    async def handle_own_pocket_note(message: Message, state: FSMContext) -> None:
        data = await state.get_data()
        amount = data["amount"]
        raw_note = message.text.strip()
        note = "" if raw_note in ("-", ".", BTN_SIN_NOTA, "Sin Nota", "sin nota") else raw_note

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

        # Notify admins
        notify_text = (
            f"👛 <b>Aviso de Bolsillo Propio</b>\n"
            f"👤 <b>{user.name}</b> avisó que puso <b>{amount:.2f} €</b> de su dinero para gastos.\n"
            f"📝 Motivo: {note or 'Sin nota'}"
        )
        await notify_admins(bot, repo, admin_telegram_ids, notify_text)

    # ── Plain text / User Registration ────────────────────────────────────

    @router.message(F.text & ~F.text.startswith("/"))
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

        role = "admin" if telegram_id in admin_telegram_ids else "trabajador"
        new_user = repo.create_user(telegram_id=telegram_id, name=name, role=role)

        role_msg = "👑 Has sido registrado como <b>Jefa / Administrador</b>." if role == "admin" else "👤 Has sido registrado como <b>Trabajador</b>."
        await message.answer(
            f"✅ Registrado como {name}.\n"
            f"{role_msg}\n\n"
            "📸 Mandame fotos de tus tickets o usá las opciones de abajo:",
            reply_markup=get_user_menu(new_user, admin_telegram_ids),
        )

    # ── Ticket Photo Processing ───────────────────────────────────────────

    @router.message(F.photo)
    async def handle_photo(message: Message) -> None:
        """Handle photo messages — process ticket image."""
        telegram_id = message.from_user.id
        user = repo.get_user_by_telegram_id(telegram_id)

        if user is None:
            await message.answer("Primero escribí /start para registrarte.")
            return

        # Get the largest photo version
        photo = message.photo[-1]

        # Download image
        file = await bot.get_file(photo.file_id)
        image_bytes = await bot.download_file(file.file_path)

        if image_bytes is None:
            return

        raw_bytes = image_bytes.read()

        # Save original image with UUID filename (security: no user-controlled paths)
        image_name = f"{uuid.uuid4().hex}.jpg"
        image_path = images_dir / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(raw_bytes)

        try:
            # Resize if too large (saves API tokens and memory), keeping aspect ratio
            resized = resize_if_large(raw_bytes, max_width=1024)
            logger.info(f"Image ready: {len(raw_bytes)} -> {len(resized)} bytes (resized)")

            # Extract data via AI
            extracted = await extractor.extract_from_image(resized)
            logger.info(
                f"Extraction: {extracted.empresa} - {extracted.total:.2f} € "
                f"(es_ticket={extracted.es_ticket})"
            )

            # Reject non-tickets and 0€ totals immediately
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

            # 1. Save to DB first (single transactional source of truth)
            repo.create_ticket(
                user_id=user.id,
                image_path=str(image_path),
                empresa=extracted.empresa,
                nif=extracted.nif,
                descripcion=extracted.descripcion,
                total=extracted.total,
                fecha=extracted.fecha,
                synced_to_excel=False,
            )

            # 2. Attempt Excel sync (catches PermissionError gracefully if open in Windows)
            synced_count, sync_ok = sync_pending_tickets(excel_path, repo)

            # 3. Compute updated balance
            bal = repo.get_user_balance(user.id)

            # 4. Build response
            nif_line = f"🪪 {extracted.nif}\n" if extracted.nif else ""
            notes: list[str] = []

            if not sync_ok:
                notes.append(
                    "⚠️ <i>Nota: El archivo Excel está abierto en el equipo. "
                    "El ticket se guardó en la base de datos y se sincronizará automáticamente cuando lo cierres.</i>"
                )

            if is_suspicious_total(extracted.total):
                notes.append(
                    f"⚠️ <i>Aviso: El importe ({extracted.total:.2f} €) parece inusualmente alto. "
                    "Verificá si coincide con el ticket.</i>"
                )

            extra_notes = ("\n\n" + "\n".join(notes)) if notes else ""
            reply_markup = get_sync_inline_keyboard() if not sync_ok else get_user_menu(user, admin_telegram_ids)

            await message.answer(
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

        except Exception as e:
            logger.error(f"Error processing ticket: {type(e).__name__}: {e}")
            await message.answer(
                "⚠️ No pude procesar ese ticket. "
                "Asegurate de que la foto sea clara y mandala de nuevo.",
                reply_markup=get_user_menu(user, admin_telegram_ids),
            )

    return router
