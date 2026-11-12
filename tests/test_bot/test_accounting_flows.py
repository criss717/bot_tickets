"""Comprehensive test suite for interactive FSM accounting flows, parsing, and notifications."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Message
from aiogram.types import User as TgUser

from bot.handlers import (
    AssignMoneyStates,
    CreateTestWorkerStates,
    OwnPocketStates,
    ReturnMoneyStates,
    create_router,
    is_admin,
    notify_admins,
    notify_worker,
    parse_euro_amount,
)
from db.models import User
from db.repository import Repository
from ocr.extractor import VisionExtractor


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "test_flows.db")


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


def make_fsm_context(storage: MemoryStorage, chat_id: int, user_id: int) -> FSMContext:
    key = StorageKey(bot_id=1, chat_id=chat_id, user_id=user_id)
    return FSMContext(storage=storage, key=key)


def get_handler_by_name(router, name: str):
    for observer in router.observers.values():
        for handler in observer.handlers:
            cb = handler.callback
            if name in getattr(cb, "__name__", ""):
                return cb
    raise ValueError(f"Handler {name} not found in router")


# ── Unit Tests: Helpers & Parsing ─────────────────────────────────────────

class TestEuroParser:
    def test_valid_amounts(self):
        assert parse_euro_amount("50") == 50.0
        assert parse_euro_amount("50.50") == 50.50
        assert parse_euro_amount("50,50") == 50.50
        assert parse_euro_amount("100 €") == 100.0
        assert parse_euro_amount("100 EUR") == 100.0
        assert parse_euro_amount(" 25,99 eur ") == 25.99

    def test_invalid_amounts(self):
        assert parse_euro_amount("0") is None
        assert parse_euro_amount("-15") is None
        assert parse_euro_amount("gratis") is None
        assert parse_euro_amount("") is None
        assert parse_euro_amount("   ") is None


class TestNotifications:
    @pytest.mark.asyncio
    async def test_notify_worker_with_real_id(self):
        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_message = AsyncMock()

        await notify_worker(mock_bot, 12345, "Test mensaje")
        mock_bot.send_message.assert_called_once_with(chat_id=12345, text="Test mensaje")

    @pytest.mark.asyncio
    async def test_notify_worker_dummy_id_skipped(self):
        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_message = AsyncMock()

        # Dummy workers have negative IDs (e.g. -1001)
        await notify_worker(mock_bot, -1001, "Test mensaje")
        mock_bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_worker_exception_handled_gracefully(self):
        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_message = AsyncMock(side_effect=Exception("Network error"))

        # Must not raise exception
        await notify_worker(mock_bot, 12345, "Test mensaje")

    @pytest.mark.asyncio
    async def test_notify_admins(self, repo: Repository):
        admin1 = repo.create_user(telegram_id=999, name="Jefa 1", role="admin")
        admin2 = repo.create_user(telegram_id=888, name="Jefa 2", role="admin")
        worker = repo.create_user(telegram_id=111, name="Pepe", role="trabajador")

        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_message = AsyncMock()

        await notify_admins(mock_bot, repo, admin_telegram_ids=(777,), text="Aviso general")

        # Must notify admin1 (999), admin2 (888) and setting admin (777), but not worker (111)
        called_chat_ids = {call.kwargs["chat_id"] for call in mock_bot.send_message.call_args_list}
        assert called_chat_ids == {999, 888, 777}


# ── FSM Flows: Asignar Dinero ─────────────────────────────────────────────

class TestAssignMoneyFSMFlow:
    @pytest.mark.asyncio
    async def test_full_assign_money_flow_to_worker(self, repo: Repository, storage: MemoryStorage, tmp_path: Path):
        admin = repo.create_user(telegram_id=999, name="La Jefa", role="admin")
        worker = repo.create_user(telegram_id=101, name="Juan Chofer", role="trabajador")

        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_message = AsyncMock()
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx", (999,))

        # 1. Admin presses "Asignar Dinero"
        h_start = get_handler_by_name(router, "handle_assign_money_start")
        msg1 = MagicMock(spec=Message)
        msg1.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        msg1.answer = AsyncMock()
        await h_start(msg1)
        assert msg1.answer.called
        assert "Elegí a quién le vas a entregar efectivo" in msg1.answer.call_args[0][0]

        # 2. Admin picks worker from inline keyboard
        state = make_fsm_context(storage, chat_id=999, user_id=999)
        h_chosen = get_handler_by_name(router, "handle_assign_worker_chosen")
        cb = MagicMock(spec=CallbackQuery)
        cb.data = f"asig_worker:{worker.id}"
        cb.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        cb.message = MagicMock()
        cb.message.edit_text = AsyncMock()
        cb.message.answer = AsyncMock()
        cb.answer = AsyncMock()

        await h_chosen(cb, state)
        assert await state.get_state() == AssignMoneyStates.waiting_for_amount.state
        data = await state.get_data()
        assert data["target_worker_id"] == worker.id
        assert data["target_worker_name"] == "Juan Chofer"

        # 3. Admin enters invalid amount
        h_amount = get_handler_by_name(router, "handle_assign_amount")
        msg_invalid = MagicMock(spec=Message)
        msg_invalid.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        msg_invalid.text = "veinte pavos"
        msg_invalid.answer = AsyncMock()

        await h_amount(msg_invalid, state)
        assert "Importe no válido" in msg_invalid.answer.call_args[0][0]
        assert await state.get_state() == AssignMoneyStates.waiting_for_amount.state

        # 4. Admin enters valid amount: 60.50 €
        msg_valid = MagicMock(spec=Message)
        msg_valid.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        msg_valid.text = "60,50 €"
        msg_valid.answer = AsyncMock()

        await h_amount(msg_valid, state)
        assert await state.get_state() == AssignMoneyStates.waiting_for_note.state
        assert (await state.get_data())["amount"] == 60.50

        # 5. Admin enters note: "Fondo ruta Valencia"
        h_note = get_handler_by_name(router, "handle_assign_note")
        msg_note = MagicMock(spec=Message)
        msg_note.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        msg_note.text = "Fondo ruta Valencia"
        msg_note.answer = AsyncMock()

        with patch("bot.handlers.sync_pending_tickets", return_value=(1, True)):
            await h_note(msg_note, state)

        # State must be cleared
        assert await state.get_state() is None

        # Movement must be recorded in SQLite
        movements = repo.get_unsynced_movements()
        assert len(movements) == 1
        mov, w_name, c_name = movements[0]
        assert mov.movement_type == "ASIGNACION"
        assert mov.amount == 60.50
        assert mov.note == "Fondo ruta Valencia"
        assert w_name == "Juan Chofer"
        assert c_name == "La Jefa"

        # Worker must have been notified via Telegram
        mock_bot.send_message.assert_called_once()
        assert mock_bot.send_message.call_args.kwargs["chat_id"] == 101
        assert "60.50 €" in mock_bot.send_message.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_assign_money_with_sin_nota_button(self, repo: Repository, storage: MemoryStorage, tmp_path: Path):
        admin = repo.create_user(telegram_id=999, name="La Jefa", role="admin")
        worker = repo.create_user(telegram_id=101, name="Juan Chofer", role="trabajador")

        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_message = AsyncMock()
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx", (999,))

        state = make_fsm_context(storage, chat_id=999, user_id=999)
        await state.update_data(target_worker_id=worker.id, target_worker_name=worker.name, amount=50.0)
        await state.set_state(AssignMoneyStates.waiting_for_note)

        h_note = get_handler_by_name(router, "handle_assign_note")
        msg = MagicMock(spec=Message)
        msg.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        msg.text = "📝 Sin Nota"
        msg.answer = AsyncMock()

        with patch("bot.handlers.sync_pending_tickets", return_value=(1, True)):
            await h_note(msg, state)

        assert await state.get_state() is None
        movements = repo.get_unsynced_movements()
        assert len(movements) == 1
        assert movements[0][0].note == ""
        assert movements[0][0].amount == 50.0


# ── FSM Flows: Crear Trabajador de Prueba ──────────────────────────────────

class TestCreateDummyWorkerFSMFlow:
    @pytest.mark.asyncio
    async def test_create_dummy_worker_flow(self, repo: Repository, storage: MemoryStorage, tmp_path: Path):
        admin = repo.create_user(telegram_id=999, name="La Jefa", role="admin")
        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx", (999,))

        state = make_fsm_context(storage, chat_id=999, user_id=999)

        # 1. Admin clicks inline button "Crear Trabajador de Prueba"
        h_dummy_start = get_handler_by_name(router, "handle_create_dummy_start")
        cb = MagicMock(spec=CallbackQuery)
        cb.data = "asig_create_dummy"
        cb.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        cb.message = MagicMock()
        cb.message.edit_text = AsyncMock()
        cb.message.answer = AsyncMock()
        cb.answer = AsyncMock()

        await h_dummy_start(cb, state)
        assert await state.get_state() == CreateTestWorkerStates.waiting_for_name.state

        # 2. Too short name validation
        h_name = get_handler_by_name(router, "handle_create_dummy_name")
        msg_short = MagicMock(spec=Message)
        msg_short.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        msg_short.text = "x"
        msg_short.answer = AsyncMock()

        await h_name(msg_short, state)
        assert "entre 2 y 100 caracteres" in msg_short.answer.call_args[0][0]
        assert await state.get_state() == CreateTestWorkerStates.waiting_for_name.state

        # 3. Valid name creation
        msg_valid = MagicMock(spec=Message)
        msg_valid.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        msg_valid.text = "Carlos Chofer Demo"
        msg_valid.answer = AsyncMock()

        await h_name(msg_valid, state)
        assert await state.get_state() is None
        assert "Carlos Chofer Demo" in msg_valid.answer.call_args[0][0]

        # Verify dummy user exists in DB with negative ID
        all_users = repo.get_all_users()
        created = [u for u in all_users if u.name == "Carlos Chofer Demo"]
        assert len(created) == 1
        assert created[0].telegram_id < 0
        assert created[0].role == "trabajador"


# ── FSM Flows: Devolver Sobrante ──────────────────────────────────────────

class TestReturnMoneyFSMFlow:
    @pytest.mark.asyncio
    async def test_full_return_money_flow(self, repo: Repository, storage: MemoryStorage, tmp_path: Path):
        admin = repo.create_user(telegram_id=999, name="La Jefa", role="admin")
        worker = repo.create_user(telegram_id=101, name="Pedro", role="trabajador")

        # Initial assignment of 100€
        repo.create_cash_movement(
            user_id=worker.id, created_by_id=admin.id,
            movement_type="ASIGNACION", amount=100.0, note="Inicial",
        )

        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_message = AsyncMock()
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx", (999,))

        state = make_fsm_context(storage, chat_id=101, user_id=101)

        # 1. Worker initiates return
        h_start = get_handler_by_name(router, "handle_return_money_start")
        msg_start = MagicMock(spec=Message)
        msg_start.from_user = TgUser(id=101, is_bot=False, first_name="Pedro")
        msg_start.answer = AsyncMock()

        await h_start(msg_start, state)
        assert await state.get_state() == ReturnMoneyStates.waiting_for_amount.state
        assert "100.00 €" in msg_start.answer.call_args[0][0]

        # 2. Worker enters amount: 35.50 €
        h_amount = get_handler_by_name(router, "handle_return_amount")
        msg_amt = MagicMock(spec=Message)
        msg_amt.from_user = TgUser(id=101, is_bot=False, first_name="Pedro")
        msg_amt.text = "35.50"
        msg_amt.answer = AsyncMock()

        await h_amount(msg_amt, state)
        assert await state.get_state() == ReturnMoneyStates.waiting_for_note.state

        # 3. Worker enters note: "Sobrante fin de semana"
        h_note = get_handler_by_name(router, "handle_return_note")
        msg_note = MagicMock(spec=Message)
        msg_note.from_user = TgUser(id=101, is_bot=False, first_name="Pedro")
        msg_note.text = "Sobrante fin de semana"
        msg_note.answer = AsyncMock()

        with patch("bot.handlers.sync_pending_tickets", return_value=(1, True)):
            await h_note(msg_note, state)

        assert await state.get_state() is None

        # Check balance: 100 - 35.50 = 64.50 €
        bal = repo.get_user_balance(worker.id)
        assert bal.net_balance == 64.50
        assert "DEBE A LA JEFA" in bal.status_label

        # Admins must have been notified
        mock_bot.send_message.assert_called_once()
        assert mock_bot.send_message.call_args.kwargs["chat_id"] == 999
        text = mock_bot.send_message.call_args.kwargs["text"]
        assert "Pedro" in text
        assert "35.50 €" in text
        assert "Sobrante fin de semana" in text


# ── FSM Flows: Puse de mi Bolsillo ────────────────────────────────────────

class TestOwnPocketFSMFlow:
    @pytest.mark.asyncio
    async def test_full_own_pocket_flow(self, repo: Repository, storage: MemoryStorage, tmp_path: Path):
        admin = repo.create_user(telegram_id=999, name="La Jefa", role="admin")
        worker = repo.create_user(telegram_id=101, name="Juan", role="trabajador")

        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_message = AsyncMock()
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx", (999,))

        state = make_fsm_context(storage, chat_id=101, user_id=101)

        # 1. Worker clicks "Puse de mi Bolsillo"
        h_start = get_handler_by_name(router, "handle_own_pocket_start")
        msg_start = MagicMock(spec=Message)
        msg_start.from_user = TgUser(id=101, is_bot=False, first_name="Juan")
        msg_start.answer = AsyncMock()

        await h_start(msg_start, state)
        assert await state.get_state() == OwnPocketStates.waiting_for_amount.state

        # 2. Worker enters amount: 50 €
        h_amt = get_handler_by_name(router, "handle_own_pocket_amount")
        msg_amt = MagicMock(spec=Message)
        msg_amt.from_user = TgUser(id=101, is_bot=False, first_name="Juan")
        msg_amt.text = "50 €"
        msg_amt.answer = AsyncMock()

        await h_amt(msg_amt, state)
        assert await state.get_state() == OwnPocketStates.waiting_for_note.state

        # 3. Worker enters note: "Fondo propio para gasolina"
        h_note = get_handler_by_name(router, "handle_own_pocket_note")
        msg_note = MagicMock(spec=Message)
        msg_note.from_user = TgUser(id=101, is_bot=False, first_name="Juan")
        msg_note.text = "Fondo propio para gasolina"
        msg_note.answer = AsyncMock()

        with patch("bot.handlers.sync_pending_tickets", return_value=(1, True)):
            await h_note(msg_note, state)

        assert await state.get_state() is None

        # Check DB movement
        movements = repo.get_unsynced_movements()
        assert len(movements) == 1
        assert movements[0][0].movement_type == "APORTE_PROPIO"
        assert movements[0][0].amount == 50.0

        # Admin notified
        mock_bot.send_message.assert_called_once()
        assert mock_bot.send_message.call_args.kwargs["chat_id"] == 999
        text = mock_bot.send_message.call_args.kwargs["text"]
        assert "Juan" in text
        assert "50.00 €" in text
        assert "de su dinero" in text


# ── Cancel Action Flows ───────────────────────────────────────────────────

class TestCancelFlows:
    @pytest.mark.asyncio
    async def test_cancel_via_text(self, repo: Repository, storage: MemoryStorage, tmp_path: Path):
        user = repo.create_user(telegram_id=123, name="Pepe", role="trabajador")
        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx")

        state = make_fsm_context(storage, chat_id=123, user_id=123)
        await state.set_state(AssignMoneyStates.waiting_for_amount)

        h_cancel = get_handler_by_name(router, "handle_cancel_text")
        msg = MagicMock(spec=Message)
        msg.from_user = TgUser(id=123, is_bot=False, first_name="Pepe")
        msg.answer = AsyncMock()

        await h_cancel(msg, state)
        assert await state.get_state() is None
        assert "Acción cancelada" in msg.answer.call_args[0][0]

    @pytest.mark.asyncio
    async def test_cancel_via_callback(self, repo: Repository, storage: MemoryStorage, tmp_path: Path):
        user = repo.create_user(telegram_id=123, name="Pepe", role="trabajador")
        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx")

        state = make_fsm_context(storage, chat_id=123, user_id=123)
        await state.set_state(AssignMoneyStates.waiting_for_amount)

        h_cancel_cb = get_handler_by_name(router, "handle_cancel_callback")
        cb = MagicMock(spec=CallbackQuery)
        cb.from_user = TgUser(id=123, is_bot=False, first_name="Pepe")
        cb.message = MagicMock()
        cb.message.delete = AsyncMock()
        cb.message.answer = AsyncMock()
        cb.answer = AsyncMock()

        await h_cancel_cb(cb, state)
        assert await state.get_state() is None
        assert "Acción cancelada" in cb.message.answer.call_args[0][0]
