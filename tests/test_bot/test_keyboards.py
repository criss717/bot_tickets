"""Tests for bot keyboards."""

from __future__ import annotations

from bot.keyboards import (
    BTN_AYUDA,
    BTN_RESUMEN,
    BTN_PANEL_JEFA,
    BTN_ASIGNAR_DINERO,
    BTN_DEVOLVER,
    BTN_BOLSILLO,
    BTN_SINCRONIZAR,
    BTN_SIN_NOTA,
    get_main_keyboard,
    get_cancel_keyboard,
    get_note_keyboard,
    get_sync_inline_keyboard,
    get_workers_inline_keyboard,
)
from db.models import User


class TestKeyboards:
    def test_worker_keyboard_structure(self):
        kb = get_main_keyboard(is_admin=False)
        assert kb.resize_keyboard is True
        assert kb.is_persistent is True
        assert "📸" in kb.input_field_placeholder

        rows = kb.keyboard
        assert len(rows) == 3
        assert rows[0][0].text == BTN_RESUMEN
        assert rows[0][1].text == BTN_DEVOLVER
        assert rows[1][0].text == BTN_BOLSILLO
        assert rows[1][1].text == BTN_SINCRONIZAR
        assert rows[2][0].text == BTN_AYUDA

    def test_admin_keyboard_structure(self):
        kb = get_main_keyboard(is_admin=True)
        assert kb.resize_keyboard is True

        rows = kb.keyboard
        assert len(rows) == 3
        assert rows[0][0].text == BTN_PANEL_JEFA
        assert rows[0][1].text == BTN_ASIGNAR_DINERO
        assert rows[1][0].text == BTN_RESUMEN
        assert rows[1][1].text == BTN_DEVOLVER
        assert rows[2][0].text == BTN_SINCRONIZAR
        assert rows[2][1].text == BTN_AYUDA

    def test_cancel_keyboard(self):
        kb = get_cancel_keyboard()
        assert kb.keyboard[0][0].text == "❌ Cancelar"

    def test_note_keyboard(self):
        kb = get_note_keyboard()
        assert kb.keyboard[0][0].text == BTN_SIN_NOTA
        assert kb.keyboard[1][0].text == "❌ Cancelar"

    def test_sync_inline_keyboard_structure(self):
        inline_kb = get_sync_inline_keyboard()
        rows = inline_kb.inline_keyboard
        assert len(rows) == 1
        button = rows[0][0]
        assert "Sincroniza" in button.text
        assert button.callback_data == "do_sync"

    def test_workers_inline_keyboard(self):
        workers = [
            User(telegram_id=1, name="Juan"),
            User(telegram_id=2, name="Pedro"),
        ]
        inline_kb = get_workers_inline_keyboard(workers)
        # Should have worker row, create dummy row, and cancel row
        assert len(inline_kb.inline_keyboard) >= 3
        assert "Juan" in inline_kb.inline_keyboard[0][0].text
        assert inline_kb.inline_keyboard[-1][0].callback_data == "cancel_action"
