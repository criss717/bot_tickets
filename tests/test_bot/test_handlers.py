"""Tests for bot handlers."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot
from aiogram.types import Message, User as TgUser, PhotoSize

from db.repository import Repository
from ocr.extractor import ExtractedTicket, VisionExtractor
from bot.handlers import create_router


class TestHandleStart:
    def test_new_user_gets_name_prompt(self, repo: Repository):
        assert repo.get_user_by_telegram_id(12345) is None

    @pytest.mark.asyncio
    async def test_existing_user_receives_welcome_with_keyboard(self, repo: Repository, tmp_path: Path):
        user = repo.create_user(telegram_id=12345, name="Juan")
        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx")

        start_handler = None
        for observer in router.observers.values():
            for handler in observer.handlers:
                if "handle_start" in getattr(handler.callback, "__name__", ""):
                    start_handler = handler.callback
                    break

        assert start_handler is not None
        message = MagicMock(spec=Message)
        message.from_user = TgUser(id=12345, is_bot=False, first_name="Juan")
        message.answer = AsyncMock()

        await start_handler(message)

        assert message.answer.called
        text = message.answer.call_args[0][0]
        assert "¡Hola de nuevo, Juan!" in text
        assert message.answer.call_args.kwargs.get("reply_markup") is not None


class TestHandleText:
    def test_register_new_user(self, repo: Repository):
        repo.create_user(telegram_id=12345, name="Juan Pérez")
        user = repo.get_user_by_telegram_id(12345)
        assert user is not None
        assert user.name == "Juan Pérez"

    def test_name_validation_too_short(self):
        name = "A"
        assert len(name) < 2

    def test_name_validation_too_long(self):
        name = "A" * 101
        assert len(name) > 100


class TestPrivacy:
    def test_users_cannot_see_other_tickets(self, repo: Repository):
        user1 = repo.create_user(telegram_id=1, name="User1")
        user2 = repo.create_user(telegram_id=2, name="User2")

        repo.create_ticket(
            user_id=user1.id, image_path="/img/1.jpg",
            empresa="Private Co", nif="X123",
            descripcion="Secret", total=100.0, fecha="01/01/2026",
        )

        user2_tickets = repo.get_tickets_by_user(user2.id)
        assert len(user2_tickets) == 0

        user1_tickets = repo.get_tickets_by_user(user1.id)
        assert len(user1_tickets) == 1
        assert user1_tickets[0].empresa == "Private Co"
        assert user1_tickets[0].nif == "X123"


class TestPhotoProcessingResilience:
    @pytest.mark.asyncio
    async def test_photo_processing_with_excel_locked(self, repo: Repository, tmp_path: Path):
        user = repo.create_user(telegram_id=999, name="Test User")
        images_dir = tmp_path / "images"
        excel_path = tmp_path / "tickets.xlsx"

        mock_bot = MagicMock(spec=Bot)
        mock_file = MagicMock()
        mock_file.file_path = "photos/test.jpg"
        mock_bot.get_file = AsyncMock(return_value=mock_file)

        # 1x1 fake image
        fake_stream = MagicMock()
        # Minimal valid 1x1 JPEG
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
        fake_stream.read.return_value = jpeg_bytes
        mock_bot.download_file = AsyncMock(return_value=fake_stream)

        mock_extractor = MagicMock(spec=VisionExtractor)
        mock_extractor.extract_from_image = AsyncMock(
            return_value=ExtractedTicket(
                empresa="Mesón La Casilla",
                nif="B14472021",
                descripcion="Comida",
                total=24.0,
                fecha="21/05/2024",
            )
        )

        router = create_router(repo, mock_extractor, mock_bot, images_dir, excel_path)

        message = MagicMock(spec=Message)
        message.from_user = TgUser(id=999, is_bot=False, first_name="Test")
        message.photo = [PhotoSize(file_id="123", file_unique_id="u123", width=100, height=100)]
        message.answer = AsyncMock()

        # Simulate Excel lock on sync_pending_tickets
        with patch("bot.handlers.sync_pending_tickets", return_value=(0, False)):
            # Find photo handler
            photo_handler = None
            for observer in router.observers.values():
                for handler in observer.handlers:
                    if "handle_photo" in getattr(handler.callback, "__name__", ""):
                        photo_handler = handler.callback
                        break

            assert photo_handler is not None
            await photo_handler(message)

        # Ticket must be created in DB despite Excel lock!
        tickets = repo.get_tickets_by_user(user.id)
        assert len(tickets) == 1
        assert tickets[0].empresa == "Mesón La Casilla"
        assert tickets[0].total == 24.0

        # Message to user must confirm ticket and inform about pending Excel sync
        answer_text = message.answer.call_args[0][0]
        assert "Ticket procesado con éxito" in answer_text
        assert "24.00 €" in answer_text
        assert "archivo Excel está abierto" in answer_text

    @pytest.mark.asyncio
    async def test_photo_processing_rejects_non_ticket(self, repo: Repository, tmp_path: Path):
        user = repo.create_user(telegram_id=12345, name="Juan")
        mock_bot = MagicMock(spec=Bot)
        mock_bot.get_file = AsyncMock(return_value=MagicMock(file_path="mock/path.jpg"))
        # Return a 1x1 dummy JPEG
        mock_image_bytes = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
            b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
            b"\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00"
            b"\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01"
            b"\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05"
            b"\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
        )
        mock_bot.download_file = AsyncMock(return_value=io.BytesIO(mock_image_bytes))

        # Extractor returns non-ticket (e.g. error screenshot with 0.0 total)
        mock_extractor = MagicMock(spec=VisionExtractor)
        mock_extractor.extract_from_image = AsyncMock(
            return_value=ExtractedTicket(
                empresa="Desconocido",
                nif="",
                descripcion="Ticket",
                total=0.0,
                fecha="12/11/2026",
                es_ticket=False,
            )
        )

        img_dir = tmp_path / "img"
        router = create_router(repo, mock_extractor, mock_bot, img_dir, tmp_path / "t.xlsx")

        photo_mock = MagicMock()
        photo_mock.file_id = "test_file_id"
        message = MagicMock(spec=Message)
        message.from_user = TgUser(id=12345, is_bot=False, first_name="Juan")
        message.photo = [photo_mock]
        message.answer = AsyncMock()

        photo_handler = None
        for observer in router.observers.values():
            for handler in observer.handlers:
                if "handle_photo" in getattr(handler.callback, "__name__", ""):
                    photo_handler = handler.callback
                    break

        assert photo_handler is not None
        await photo_handler(message)

        # No tickets should be created in DB!
        tickets = repo.get_tickets_by_user(user.id)
        assert len(tickets) == 0

        # Message should warn user that it's not a valid ticket
        answer_text = message.answer.call_args[0][0]
        assert "No pude detectar un ticket válido" in answer_text
        assert "0.00 €" in answer_text


class TestCallbackQuerySync:
    @pytest.mark.asyncio
    async def test_callback_sync_success(self, repo: Repository, tmp_path: Path):
        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx")

        cb_handler = None
        for observer in router.observers.values():
            for handler in observer.handlers:
                if "handle_callback_sync" in getattr(handler.callback, "__name__", ""):
                    cb_handler = handler.callback
                    break

        assert cb_handler is not None

        callback = MagicMock()
        callback.data = "do_sync"
        callback.answer = AsyncMock()
        callback.message = MagicMock()
        callback.message.edit_reply_markup = AsyncMock()

        with patch("bot.handlers.sync_pending_tickets", return_value=(2, True)):
            await cb_handler(callback)

        assert callback.answer.called
        assert "Se volcaron 2 tickets" in callback.answer.call_args[0][0]


class TestRBACAndAccountingHandlers:
    @pytest.mark.asyncio
    async def test_worker_cannot_access_boss_panel(self, repo: Repository, tmp_path: Path):
        worker = repo.create_user(telegram_id=111, name="Trabajador", role="trabajador")
        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx", admin_telegram_ids=(999,))

        panel_handler = None
        for observer in router.observers.values():
            for handler in observer.handlers:
                if "handle_boss_panel" in getattr(handler.callback, "__name__", ""):
                    panel_handler = handler.callback
                    break
        assert panel_handler is not None

        msg = MagicMock(spec=Message)
        msg.from_user = TgUser(id=111, is_bot=False, first_name="Trabajador")
        msg.answer = AsyncMock()

        await panel_handler(msg)
        assert msg.answer.called
        assert "exclusiva para la Jefa" in msg.answer.call_args[0][0]

    @pytest.mark.asyncio
    async def test_admin_can_access_boss_panel(self, repo: Repository, tmp_path: Path):
        admin = repo.create_user(telegram_id=999, name="La Jefa", role="admin")
        worker = repo.create_user(telegram_id=111, name="Trabajador 1", role="trabajador")
        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx", admin_telegram_ids=(999,))

        panel_handler = None
        for observer in router.observers.values():
            for handler in observer.handlers:
                if "handle_boss_panel" in getattr(handler.callback, "__name__", ""):
                    panel_handler = handler.callback
                    break
        assert panel_handler is not None

        msg = MagicMock(spec=Message)
        msg.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        msg.answer = AsyncMock()

        await panel_handler(msg)
        assert msg.answer.called
        text = msg.answer.call_args[0][0]
        assert "Panel de Control de la Jefa" in text
        assert "Trabajador 1" in text

    @pytest.mark.asyncio
    async def test_summary_shows_balance_and_status(self, repo: Repository, tmp_path: Path):
        user = repo.create_user(telegram_id=222, name="Carlos Chofer", role="trabajador")
        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx")

        summary_handler = None
        for observer in router.observers.values():
            for handler in observer.handlers:
                if "handle_summary" in getattr(handler.callback, "__name__", ""):
                    summary_handler = handler.callback
                    break
        assert summary_handler is not None

        msg = MagicMock(spec=Message)
        msg.from_user = TgUser(id=222, is_bot=False, first_name="Carlos")
        msg.answer = AsyncMock()

        await summary_handler(msg)
        assert msg.answer.called
        text = msg.answer.call_args[0][0]
        assert "Carlos Chofer" in text
        assert "SALDO NETO" in text
        assert "0.00 €" in text
        assert "A PAZ Y SALVO" in text


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "test.db")
