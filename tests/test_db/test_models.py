"""Tests for DB models."""

from __future__ import annotations

from datetime import datetime

from db.models import Ticket, User


class TestUserModel:
    """Tests for the User dataclass."""

    def test_user_creation_minimal(self):
        user = User(telegram_id=12345, name="Juan")
        assert user.telegram_id == 12345
        assert user.name == "Juan"
        assert user.id is None
        assert isinstance(user.created_at, datetime)

    def test_user_creation_full(self):
        now = datetime(2026, 1, 15, 10, 30)
        user = User(telegram_id=1, name="Ana", created_at=now, id=42)
        assert user.id == 42
        assert user.created_at == now


class TestTicketModel:
    """Tests for the Ticket dataclass."""

    def test_ticket_creation_minimal(self):
        ticket = Ticket(
            user_id=1,
            image_path="/img/ticket.jpg",
            empresa="Repsol",
            nif="B12345678",
            descripcion="Gasolina 95",
            total=36.89,
            fecha="15/08/2026",
        )
        assert ticket.user_id == 1
        assert ticket.empresa == "Repsol"
        assert ticket.nif == "B12345678"
        assert ticket.total == 36.89
        assert ticket.id is None

    def test_ticket_creation_without_nif(self):
        ticket = Ticket(
            user_id=1,
            image_path="/img/t.jpg",
            empresa="Bar Pepe",
            nif="",
            descripcion="Café",
            total=1.50,
            fecha="18/08/2026",
        )
        assert ticket.nif == ""
