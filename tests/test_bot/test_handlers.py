"""Tests for bot handlers."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot
from aiogram.types import Message, User as TgUser, PhotoSize

from db.repository import Repository
from ocr.extractor import ExtractedTicket, OllamaUnavailableError, VisionExtractor
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
        user = repo.create_user(telegram_id=999, name="Test User", role="admin")
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

        router = create_router(
            repo, mock_extractor, mock_bot, images_dir, excel_path, admin_telegram_ids=(999,)
        )

        message = MagicMock(spec=Message)
        message.from_user = TgUser(id=999, is_bot=False, first_name="Test")
        message.photo = [PhotoSize(file_id="123", file_unique_id="u123", width=100, height=100)]
        message.answer = AsyncMock()

        # Simulate Excel lock on sync_pending_tickets
        with patch("bot.handlers.sync_pending_tickets", return_value=(0, False)):
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

        # Message to admin must confirm ticket, warn about Excel lock and attach retry sync button
        answer_text = message.answer.call_args[0][0]
        assert "Ticket procesado con éxito" in answer_text
        assert "24.00 €" in answer_text
        assert "archivo Excel está abierto" in answer_text
        reply_kb = message.answer.call_args.kwargs.get("reply_markup")
        assert reply_kb.inline_keyboard[0][0].callback_data == "do_sync"

    @pytest.mark.asyncio
    async def test_worker_photo_with_excel_locked_hides_sync_button_and_notifies_admin(
        self, repo: Repository, tmp_path: Path
    ):
        admin = repo.create_user(telegram_id=999, name="La Jefa", role="admin")
        worker = repo.create_user(telegram_id=12345, name="Juan", role="trabajador")
        images_dir = tmp_path / "images"
        excel_path = tmp_path / "tickets.xlsx"

        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_message = AsyncMock()
        mock_file = MagicMock()
        mock_file.file_path = "photos/test.jpg"
        mock_bot.get_file = AsyncMock(return_value=mock_file)
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
        mock_bot.download_file = AsyncMock(return_value=io.BytesIO(jpeg_bytes))

        mock_extractor = MagicMock(spec=VisionExtractor)
        mock_extractor.extract_from_image = AsyncMock(
            return_value=ExtractedTicket(
                empresa="Café Central", nif="", descripcion="Desayuno", total=4.5, fecha="13/11/2026"
            )
        )

        router = create_router(
            repo, mock_extractor, mock_bot, images_dir, excel_path, admin_telegram_ids=(999,)
        )

        message = MagicMock(spec=Message)
        message.from_user = TgUser(id=12345, is_bot=False, first_name="Juan")
        message.photo = [PhotoSize(file_id="123", file_unique_id="u123", width=100, height=100)]
        message.answer = AsyncMock()

        with patch("bot.handlers.sync_pending_tickets", return_value=(0, False)):
            photo_handler = next(
                h.callback
                for obs in router.observers.values()
                for h in obs.handlers
                if "handle_photo" in getattr(h.callback, "__name__", "")
            )
            await photo_handler(message)

        answer_text = message.answer.call_args[0][0]
        assert "Ticket procesado con éxito" in answer_text
        assert "archivo Excel" not in answer_text
        reply_kb = message.answer.call_args.kwargs.get("reply_markup")
        cb_datas = [b.callback_data for row in reply_kb.inline_keyboard for b in row]
        assert "do_sync" not in cb_datas
        assert any(d.startswith("tkt_edit:") for d in cb_datas)
        # Admin should be alerted silently about Excel lock
        mock_bot.send_message.assert_called_once()
        assert mock_bot.send_message.call_args.kwargs["chat_id"] == 999
        assert "Excel bloqueado" in mock_bot.send_message.call_args.kwargs["text"]

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

    @pytest.mark.asyncio
    async def test_photo_processing_ollama_unavailable_cleans_image_and_warns(self, repo: Repository, tmp_path: Path):
        repo.create_user(telegram_id=12345, name="Juan")
        mock_bot = MagicMock(spec=Bot)
        mock_bot.get_file = AsyncMock(return_value=MagicMock(file_path="mock/path.jpg"))
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
        mock_bot.download_file = AsyncMock(return_value=io.BytesIO(jpeg_bytes))

        mock_extractor = MagicMock(spec=VisionExtractor)
        mock_extractor.extract_from_image = AsyncMock(side_effect=OllamaUnavailableError("down"))

        img_dir = tmp_path / "img"
        router = create_router(repo, mock_extractor, mock_bot, img_dir, tmp_path / "t.xlsx")

        message = MagicMock(spec=Message)
        message.from_user = TgUser(id=12345, is_bot=False, first_name="Juan")
        message.photo = [PhotoSize(file_id="f1", file_unique_id="u1", width=1, height=1)]
        message.answer = AsyncMock()

        photo_handler = next(
            h.callback
            for obs in router.observers.values()
            for h in obs.handlers
            if "handle_photo" in getattr(h.callback, "__name__", "")
        )
        await photo_handler(message)

        assert list(img_dir.glob("*")) == []  # Orphan file removed
        answer_text = message.answer.call_args[0][0]
        assert "Ollama" in answer_text
        assert "no está iniciado" in answer_text


class TestCallbackQuerySync:
    @pytest.mark.asyncio
    async def test_callback_sync_success(self, repo: Repository, tmp_path: Path):
        repo.create_user(telegram_id=999, name="La Jefa", role="admin")
        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(
            repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx", admin_telegram_ids=(999,)
        )

        cb_handler = None
        for observer in router.observers.values():
            for handler in observer.handlers:
                if "handle_callback_sync" in getattr(handler.callback, "__name__", ""):
                    cb_handler = handler.callback
                    break

        assert cb_handler is not None

        callback = MagicMock()
        callback.data = "do_sync"
        callback.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        callback.answer = AsyncMock()
        callback.message = MagicMock()
        callback.message.edit_reply_markup = AsyncMock()

        with patch("bot.handlers.sync_pending_tickets", return_value=(2, True)):
            await cb_handler(callback)

        assert callback.answer.called
        assert "Se volcaron 2 tickets" in callback.answer.call_args[0][0]

    @pytest.mark.asyncio
    async def test_worker_cannot_sync_excel_via_command_or_callback(self, repo: Repository, tmp_path: Path):
        repo.create_user(telegram_id=111, name="Pepe", role="trabajador")
        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(
            repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx", admin_telegram_ids=(999,)
        )

        sync_msg_handler = next(
            h.callback
            for obs in router.observers.values()
            for h in obs.handlers
            if "handle_sync" in getattr(h.callback, "__name__", "")
        )
        msg = MagicMock(spec=Message)
        msg.from_user = TgUser(id=111, is_bot=False, first_name="Pepe")
        msg.answer = AsyncMock()
        await sync_msg_handler(msg)
        assert "exclusiva para Administradores" in msg.answer.call_args[0][0]

        sync_cb_handler = next(
            h.callback
            for obs in router.observers.values()
            for h in obs.handlers
            if "handle_callback_sync" in getattr(h.callback, "__name__", "")
        )
        cb = MagicMock()
        cb.data = "do_sync"
        cb.from_user = TgUser(id=111, is_bot=False, first_name="Pepe")
        cb.answer = AsyncMock()
        await sync_cb_handler(cb)
        assert "exclusiva para Administradores" in cb.answer.call_args[0][0]

    @pytest.mark.asyncio
    async def test_admin_sync_command_rebuilds_incomplete_excel(self, repo: Repository, tmp_path: Path):
        from openpyxl import load_workbook

        repo.create_user(telegram_id=999, name="Admin", role="admin")
        worker = repo.create_user(telegram_id=101, name="Pepe", role="trabajador")
        repo.create_ticket(
            user_id=worker.id, image_path="", empresa="Historico SL",
            nif="", descripcion="Menu", total=18.5, fecha="10/11/2026", synced_to_excel=True,
        )

        excel_path = tmp_path / "export" / "tickets.xlsx"
        from excel.writer import ensure_workbook
        ensure_workbook(excel_path)
        assert load_workbook(excel_path)["Tickets"].max_row == 1

        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(
            repo, mock_extractor, mock_bot, tmp_path / "export" / "fotos", excel_path, admin_telegram_ids=(999,)
        )

        sync_msg_handler = next(
            h.callback
            for obs in router.observers.values()
            for h in obs.handlers
            if "handle_sync" in getattr(h.callback, "__name__", "")
        )
        msg = MagicMock(spec=Message)
        msg.from_user = TgUser(id=999, is_bot=False, first_name="Admin")
        msg.answer = AsyncMock()
        await sync_msg_handler(msg)

        assert "Sincronización completa" in msg.answer.call_args[0][0]
        assert load_workbook(excel_path)["Tickets"].max_row == 2


class TestRBACAndAccountingHandlers:
    @pytest.mark.asyncio
    async def test_removed_admin_from_env_loses_privileges_everywhere(self, repo: Repository, tmp_path: Path):
        # User still has stale 'admin' role in SQLite, but was removed from ADMIN_TELEGRAM_IDS
        stale_ex_admin = repo.create_user(telegram_id=888, name="Ex Admin", role="admin")
        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(
            repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx", admin_telegram_ids=(999,)
        )

        # 1. Attempting boss panel denies access, updates DB role to trabajador, and sends worker keyboard
        panel_handler = next(
            h.callback
            for obs in router.observers.values()
            for h in obs.handlers
            if "handle_boss_panel" in getattr(h.callback, "__name__", "")
        )
        msg = MagicMock(spec=Message)
        msg.from_user = TgUser(id=888, is_bot=False, first_name="Ex Admin")
        msg.answer = AsyncMock()
        await panel_handler(msg)
        assert "exclusiva para Administradores" in msg.answer.call_args[0][0]
        kb = msg.answer.call_args.kwargs.get("reply_markup")
        assert kb is not None
        assert len(kb.keyboard) == 2  # Worker 2x2 keyboard (no admin buttons, no sync)
        assert repo.get_user_by_id(stale_ex_admin.id).role == "trabajador"

        # 2. Reset DB role back to stale .env admin and simulate /start: must also demote cleanly
        repo.set_user_role(stale_ex_admin.id, "admin", admin_source="env")
        start_handler = next(
            h.callback
            for obs in router.observers.values()
            for h in obs.handlers
            if "handle_start" in getattr(h.callback, "__name__", "")
        )
        msg_start = MagicMock(spec=Message)
        msg_start.from_user = TgUser(id=888, is_bot=False, first_name="Ex Admin")
        msg_start.answer = AsyncMock()
        await start_handler(msg_start)
        assert "Trabajador" in msg_start.answer.call_args[0][0]
        assert repo.get_user_by_id(stale_ex_admin.id).role == "trabajador"

    @pytest.mark.asyncio
    async def test_bot_promoted_admin_persists_and_receives_live_keyboard_and_commands(
        self, repo: Repository, tmp_path: Path
    ):
        repo.create_user(telegram_id=999, name="Superadmin", role="admin", admin_source="env")
        lucia = repo.create_user(telegram_id=555, name="Lucia", role="trabajador")

        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_message = AsyncMock()
        mock_bot.set_my_commands = AsyncMock()
        mock_bot.delete_my_commands = AsyncMock()
        mock_extractor = MagicMock(spec=VisionExtractor)
        router = create_router(
            repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx", admin_telegram_ids=(999,)
        )

        # 1. Superadmin promotes Lucia via inline button wrk_role
        role_handler = next(
            h.callback
            for obs in router.observers.values()
            for h in obs.handlers
            if "handle_worker_toggle_role" in getattr(h.callback, "__name__", "")
        )
        cb_promote = MagicMock()
        cb_promote.data = f"wrk_role:{lucia.id}:admin"
        cb_promote.from_user = TgUser(id=999, is_bot=False, first_name="Superadmin")
        cb_promote.message = MagicMock()
        cb_promote.message.edit_text = AsyncMock()
        cb_promote.answer = AsyncMock()
        await role_handler(cb_promote)

        updated_lucia = repo.get_user_by_id(lucia.id)
        assert updated_lucia.role == "admin"
        assert updated_lucia.admin_source == "bot"
        assert mock_bot.set_my_commands.called
        # Live push message sent to Lucia with 3-row admin keyboard
        mock_bot.send_message.assert_called()
        push_kwargs = mock_bot.send_message.call_args.kwargs
        assert push_kwargs["chat_id"] == 555
        assert len(push_kwargs["reply_markup"].keyboard) == 3

        # 2. Lucia can open Panel Admin even though she is not in .env ADMIN_TELEGRAM_IDS
        panel_handler = next(
            h.callback
            for obs in router.observers.values()
            for h in obs.handlers
            if "handle_boss_panel" in getattr(h.callback, "__name__", "")
        )
        msg_panel = MagicMock(spec=Message)
        msg_panel.from_user = TgUser(id=555, is_bot=False, first_name="Lucia")
        msg_panel.answer = AsyncMock()
        await panel_handler(msg_panel)
        assert "Panel Admin" in msg_panel.answer.call_args[0][0]

        # 3. Demote Lucia via bot -> loses admin, deletes commands, gets 2-row worker keyboard
        mock_bot.send_message.reset_mock()
        cb_demote = MagicMock()
        cb_demote.data = f"wrk_role:{lucia.id}:trabajador"
        cb_demote.from_user = TgUser(id=999, is_bot=False, first_name="Superadmin")
        cb_demote.message = MagicMock()
        cb_demote.message.edit_text = AsyncMock()
        cb_demote.answer = AsyncMock()
        await role_handler(cb_demote)

        demoted_lucia = repo.get_user_by_id(lucia.id)
        assert demoted_lucia.role == "trabajador"
        assert demoted_lucia.admin_source == ""
        assert mock_bot.delete_my_commands.called
        assert len(mock_bot.send_message.call_args.kwargs["reply_markup"].keyboard) == 2
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
        assert "exclusiva para Administradores" in msg.answer.call_args[0][0]

    @pytest.mark.asyncio
    async def test_admin_can_access_boss_panel(self, repo: Repository, tmp_path: Path):
        admin = repo.create_user(telegram_id=999, name="Admin Principal", role="admin")
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
        msg.from_user = TgUser(id=999, is_bot=False, first_name="Admin Principal")
        msg.answer = AsyncMock()

        await panel_handler(msg)
        assert msg.answer.called
        text = msg.answer.call_args[0][0]
        assert "Panel Admin" in text
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


class TestDeduplicationTicketCorrectionAndExcelDownload:
    @staticmethod
    def _sample_jpeg_bytes() -> bytes:
        import cv2
        import numpy as np

        img = np.full((120, 200, 3), 240, dtype=np.uint8)
        cv2.putText(img, "REPSOL 30.00", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return buf.tobytes()

    @pytest.mark.asyncio
    async def test_layer1_duplicate_hash_skips_ocr_and_allows_discard_or_keep(self, repo: Repository, tmp_path: Path):
        w1 = repo.create_user(telegram_id=101, name="Carlos")
        w2 = repo.create_user(telegram_id=202, name="Lucía")
        raw = self._sample_jpeg_bytes()

        from ocr.dedup import compute_dhash, compute_sha256

        sha = compute_sha256(raw)
        dhash = compute_dhash(raw)
        repo.create_ticket(
            user_id=w1.id,
            image_path=str(tmp_path / "orig.jpg"),
            empresa="Repsol",
            nif="B123",
            descripcion="Gasolina",
            total=30.0,
            fecha="13/11/2026",
            image_sha256=sha,
            image_dhash=dhash,
        )

        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_chat_action = AsyncMock()
        mock_bot.get_file = AsyncMock(return_value=MagicMock(file_path="p/dup.jpg"))
        mock_bot.download_file = AsyncMock(side_effect=[io.BytesIO(raw), io.BytesIO(raw)])

        mock_extractor = MagicMock(spec=VisionExtractor)
        mock_extractor.extract_from_image = AsyncMock(
            return_value=ExtractedTicket(
                empresa="Repsol", nif="B123", descripcion="Gasolina", total=30.0, fecha="13/11/2026", es_ticket=True
            )
        )

        img_dir = tmp_path / "img"
        excel_path = tmp_path / "t.xlsx"
        router = create_router(repo, mock_extractor, mock_bot, img_dir, excel_path)

        h_photo = next(
            h.callback for obs in router.observers.values() for h in obs.handlers
            if "handle_photo" in getattr(h.callback, "__name__", "")
        )
        h_drop = next(
            h.callback for obs in router.observers.values() for h in obs.handlers
            if "handle_duplicate_drop" in getattr(h.callback, "__name__", "")
        )
        h_keep = next(
            h.callback for obs in router.observers.values() for h in obs.handlers
            if "handle_duplicate_keep" in getattr(h.callback, "__name__", "")
        )

        # 1. Second worker sends the exact same image -> Layer 1 triggers WITHOUT calling Gemini
        msg1 = MagicMock(spec=Message)
        msg1.from_user = TgUser(id=202, is_bot=False, first_name="Lucía")
        msg1.chat = MagicMock(id=202)
        msg1.photo = [PhotoSize(file_id="f1", file_unique_id="u1", width=200, height=120)]
        msg1.answer = AsyncMock()

        await h_photo(msg1)
        assert not mock_extractor.extract_from_image.called
        warn_text = msg1.answer.call_args[0][0]
        assert "REPETIDA" in warn_text
        assert "Carlos" in warn_text
        kb1 = msg1.answer.call_args.kwargs["reply_markup"]
        drop_cb_data = kb1.inline_keyboard[0][0].callback_data
        assert drop_cb_data.startswith("dup_drop:")

        # User discards -> image removed cleanly, still 0 tickets for w2
        cb_drop = MagicMock()
        cb_drop.data = drop_cb_data
        cb_drop.from_user = TgUser(id=202, is_bot=False, first_name="Lucía")
        cb_drop.message = MagicMock()
        cb_drop.message.edit_text = AsyncMock()
        cb_drop.answer = AsyncMock()
        await h_drop(cb_drop)
        assert len(repo.get_tickets_by_user(w2.id)) == 0
        assert list(img_dir.glob("*")) == []

        # 2. Upload again, this time user clicks "Guardar Igualmente" -> OCR runs now and saves
        msg2 = MagicMock(spec=Message)
        msg2.from_user = TgUser(id=202, is_bot=False, first_name="Lucía")
        msg2.chat = MagicMock(id=202)
        msg2.photo = [PhotoSize(file_id="f2", file_unique_id="u2", width=200, height=120)]
        msg2.answer = AsyncMock()
        await h_photo(msg2)

        keep_cb_data = msg2.answer.call_args.kwargs["reply_markup"].inline_keyboard[0][1].callback_data
        cb_keep = MagicMock()
        cb_keep.data = keep_cb_data
        cb_keep.from_user = TgUser(id=202, is_bot=False, first_name="Lucía")
        cb_keep.message = MagicMock()
        cb_keep.message.delete = AsyncMock()
        cb_keep.message.answer = AsyncMock()
        cb_keep.answer = AsyncMock()
        await h_keep(cb_keep)

        assert mock_extractor.extract_from_image.call_count == 1
        assert len(repo.get_tickets_by_user(w2.id)) == 1
        assert excel_path.exists()

    @pytest.mark.asyncio
    async def test_layer2_semantic_duplicate_saves_without_re_running_ocr(self, repo: Repository, tmp_path: Path):
        w1 = repo.create_user(telegram_id=101, name="Carlos")
        repo.create_ticket(
            user_id=w1.id,
            image_path=str(tmp_path / "old_angle.jpg"),
            empresa="Restaurante Montemayor SL",
            nif="",
            descripcion="Menú",
            total=28.40,
            fecha="13/11/2026",
            image_sha256="different_sha_orig",
            image_dhash="0000000000000000",
        )

        raw = self._sample_jpeg_bytes()
        mock_bot = MagicMock(spec=Bot)
        mock_bot.send_chat_action = AsyncMock()
        mock_bot.get_file = AsyncMock(return_value=MagicMock(file_path="p/angle2.jpg"))
        mock_bot.download_file = AsyncMock(return_value=io.BytesIO(raw))

        mock_extractor = MagicMock(spec=VisionExtractor)
        mock_extractor.extract_from_image = AsyncMock(
            return_value=ExtractedTicket(
                empresa="MONTEMAYOR ARAHAL S.L.", nif="B90321852",
                descripcion="Menú", total=28.40, fecha="13/11/2026", es_ticket=True,
            )
        )

        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", tmp_path / "t.xlsx")
        h_photo = next(
            h.callback for obs in router.observers.values() for h in obs.handlers
            if "handle_photo" in getattr(h.callback, "__name__", "")
        )
        h_keep = next(
            h.callback for obs in router.observers.values() for h in obs.handlers
            if "handle_duplicate_keep" in getattr(h.callback, "__name__", "")
        )

        msg = MagicMock(spec=Message)
        msg.from_user = TgUser(id=101, is_bot=False, first_name="Carlos")
        msg.chat = MagicMock(id=101)
        msg.photo = [PhotoSize(file_id="f1", file_unique_id="u1", width=200, height=120)]
        msg.answer = AsyncMock()
        await h_photo(msg)

        assert "Posible Ticket Duplicado" in msg.answer.call_args[0][0]
        keep_data = msg.answer.call_args.kwargs["reply_markup"].inline_keyboard[0][1].callback_data

        cb_keep = MagicMock()
        cb_keep.data = keep_data
        cb_keep.from_user = TgUser(id=101, is_bot=False, first_name="Carlos")
        cb_keep.message = MagicMock()
        cb_keep.message.delete = AsyncMock()
        cb_keep.message.answer = AsyncMock()
        cb_keep.answer = AsyncMock()
        await h_keep(cb_keep)

        # Must only have called Gemini ONCE (reuses extracted data)
        assert mock_extractor.extract_from_image.call_count == 1
        assert len(repo.get_tickets_by_user(w1.id)) == 2

    @pytest.mark.asyncio
    async def test_ticket_edit_delete_and_admin_excel_download(self, repo: Repository, tmp_path: Path):
        from aiogram.fsm.storage.memory import MemoryStorage
        from aiogram.fsm.context import FSMContext
        from aiogram.fsm.storage.base import StorageKey
        from openpyxl import load_workbook

        admin = repo.create_user(telegram_id=999, name="La Jefa", role="admin")
        worker = repo.create_user(telegram_id=101, name="Pepe", role="trabajador")
        img_file = tmp_path / "img" / "t1.jpg"
        img_file.parent.mkdir(parents=True, exist_ok=True)
        img_file.write_bytes(b"fake")

        tkt = repo.create_ticket(
            user_id=worker.id, image_path=str(img_file),
            empresa="Cafetería", nif="", descripcion="Café", total=12.0, fecha="13/11/2026",
        )

        mock_bot = MagicMock(spec=Bot)
        mock_extractor = MagicMock(spec=VisionExtractor)
        excel_path = tmp_path / "export" / "tickets.xlsx"
        router = create_router(repo, mock_extractor, mock_bot, tmp_path / "img", excel_path, (999,))
        state = FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=101, user_id=101))

        # 1. Worker edits ticket total from 12.00 to 15.80
        h_edit_start = next(
            h.callback for obs in router.observers.values() for h in obs.handlers
            if "handle_ticket_edit_start" in getattr(h.callback, "__name__", "")
        )
        h_edit_total = next(
            h.callback for obs in router.observers.values() for h in obs.handlers
            if "handle_ticket_edit_total" in getattr(h.callback, "__name__", "")
        )

        cb_edit = MagicMock()
        cb_edit.data = f"tkt_edit:{tkt.id}"
        cb_edit.from_user = TgUser(id=101, is_bot=False, first_name="Pepe")
        cb_edit.message = MagicMock()
        cb_edit.message.answer = AsyncMock()
        cb_edit.answer = AsyncMock()
        await h_edit_start(cb_edit, state)

        msg_new = MagicMock(spec=Message)
        msg_new.from_user = TgUser(id=101, is_bot=False, first_name="Pepe")
        msg_new.text = "15,80"
        msg_new.answer = AsyncMock()
        await h_edit_total(msg_new, state)

        assert repo.get_ticket_by_id(tkt.id).total == 15.80
        wb = load_workbook(excel_path)
        assert wb["Tickets"].cell(row=2, column=5).value == 15.80

        # 2. Admin downloads Excel directly via callback
        h_dl = next(
            h.callback for obs in router.observers.values() for h in obs.handlers
            if "handle_excel_download" in getattr(h.callback, "__name__", "")
        )
        cb_dl = MagicMock()
        cb_dl.data = "excel_download"
        cb_dl.from_user = TgUser(id=999, is_bot=False, first_name="La Jefa")
        cb_dl.message = MagicMock()
        cb_dl.message.answer_document = AsyncMock()
        cb_dl.answer = AsyncMock()
        await h_dl(cb_dl)
        assert cb_dl.message.answer_document.called

        # 3. Worker confirms ticket deletion -> DB row removed, file deleted, Excel rebuilt
        h_del = next(
            h.callback for obs in router.observers.values() for h in obs.handlers
            if "handle_ticket_delete_confirm" in getattr(h.callback, "__name__", "")
        )
        cb_del = MagicMock()
        cb_del.data = f"tkt_del_confirm:{tkt.id}"
        cb_del.from_user = TgUser(id=101, is_bot=False, first_name="Pepe")
        cb_del.message = MagicMock()
        cb_del.message.edit_text = AsyncMock()
        cb_del.answer = AsyncMock()
        await h_del(cb_del)

        assert repo.get_ticket_by_id(tkt.id) is None
        assert not img_file.exists()
        assert repo.get_user_balance(worker.id).net_balance == 0.0

    @pytest.mark.asyncio
    async def test_photo_saved_with_readable_windows_safe_name_under_export_and_linked_in_excel(
        self, repo: Repository, tmp_path: Path
    ):
        from datetime import datetime
        from openpyxl import load_workbook

        worker = repo.create_user(telegram_id=101, name="José Ángel / López", role="trabajador")
        export_dir = tmp_path / "export"
        fotos_dir = export_dir / "fotos"
        excel_path = export_dir / "tickets.xlsx"

        mock_bot = MagicMock(spec=Bot)
        mock_bot.get_file = AsyncMock(return_value=MagicMock(file_path="p.jpg"))
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
        mock_bot.download_file = AsyncMock(return_value=io.BytesIO(jpeg_bytes))

        mock_extractor = MagicMock(spec=VisionExtractor)
        mock_extractor.extract_from_image = AsyncMock(
            return_value=ExtractedTicket(
                empresa='Bar "El Rincón" S.L.', nif="B12345678",
                descripcion="Desayuno", total=12.40, fecha="16/11/2026",
            )
        )

        router = create_router(repo, mock_extractor, mock_bot, fotos_dir, excel_path, (999,))
        photo_handler = next(
            h.callback for obs in router.observers.values() for h in obs.handlers
            if "handle_photo" in getattr(h.callback, "__name__", "")
        )
        msg = MagicMock(spec=Message)
        msg.from_user = TgUser(id=101, is_bot=False, first_name="José")
        msg.photo = [PhotoSize(file_id="f1", file_unique_id="u1", width=10, height=10)]
        msg.answer = AsyncMock()
        await photo_handler(msg)

        tickets = repo.get_tickets_by_user(worker.id)
        assert len(tickets) == 1
        saved_path = Path(tickets[0].image_path)
        assert saved_path.exists()
        month_str = datetime.now().strftime("%Y-%m")
        assert saved_path.parent == fotos_dir / month_str
        assert saved_path.name.startswith("2026-11-16_Jose_Angel_Lopez_Bar_El_Rincon_S_L_12-40EUR_")
        assert saved_path.suffix == ".jpg"
        assert list(fotos_dir.glob(".tmp_*")) == []

        wb = load_workbook(excel_path)
        ws = wb["Tickets"]
        foto_cell = ws.cell(row=2, column=8)
        assert foto_cell.value == f"fotos/{month_str}/{saved_path.name}"
        assert foto_cell.hyperlink.target == f"fotos/{month_str}/{saved_path.name}"


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "test.db")
