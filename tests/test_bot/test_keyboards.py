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
    get_amount_inline_keyboard,
    get_boss_panel_inline_keyboard,
    get_ticket_actions_inline_keyboard,
    get_duplicate_decision_keyboard,
)
from db.models import User


class TestKeyboards:
    def test_worker_keyboard_structure(self):
        kb = get_main_keyboard(is_admin=False)
        assert kb.resize_keyboard is True
        assert kb.is_persistent is True
        assert "📸" in kb.input_field_placeholder

        rows = kb.keyboard
        assert len(rows) == 2
        assert rows[0][0].text == BTN_RESUMEN
        assert rows[0][1].text == BTN_DEVOLVER
        assert rows[1][0].text == BTN_BOLSILLO
        assert rows[1][1].text == BTN_AYUDA
        all_texts = [btn.text for row in rows for btn in row]
        assert BTN_SINCRONIZAR not in all_texts

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

    def test_quick_amount_and_admin_inline_keyboards(self):
        amt_kb = get_amount_inline_keyboard(
            "ret_amt",
            amounts=(20, 50, 100),
            extra_top_button=("💸 Devolver todo (35.00 €)", "ret_amt:35.00"),
        )
        assert amt_kb.inline_keyboard[0][0].callback_data == "ret_amt:35.00"
        assert amt_kb.inline_keyboard[1][0].callback_data == "ret_amt:20"
        assert amt_kb.inline_keyboard[1][1].callback_data == "ret_amt:50"
        assert amt_kb.inline_keyboard[1][2].callback_data == "ret_amt:100"
        assert amt_kb.inline_keyboard[-1][0].callback_data == "cancel_action"

        boss_kb = get_boss_panel_inline_keyboard()
        cb_datas = [btn.callback_data for row in boss_kb.inline_keyboard for btn in row]
        assert "excel_download" in cb_datas
        assert "admin_unassign_menu" in cb_datas
        assert "admin_tickets_menu" in cb_datas
        assert "admin_workers_menu" in cb_datas

        tkt_kb = get_ticket_actions_inline_keyboard(42)
        assert [b.callback_data for b in tkt_kb.inline_keyboard[0]] == ["tkt_edit:42", "tkt_del:42"]

        dup_kb = get_duplicate_decision_keyboard("abc123")
        assert [b.callback_data for b in dup_kb.inline_keyboard[0]] == ["dup_drop:abc123", "dup_keep:abc123"]
