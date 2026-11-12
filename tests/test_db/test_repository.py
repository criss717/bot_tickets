"""Tests for DB repository."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from db.models import User, Ticket
from db.repository import Repository


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    """Create a fresh repository with a temp database."""
    db_path = tmp_path / "test.db"
    return Repository(db_path)


@pytest.fixture
def sample_user(repo: Repository) -> User:
    """Create a sample user in the repository."""
    return repo.create_user(telegram_id=12345, name="Juan Pérez")


class TestCreateUser:
    def test_create_user(self, repo: Repository):
        user = repo.create_user(telegram_id=100, name="Ana")
        assert user.id is not None
        assert user.telegram_id == 100
        assert user.name == "Ana"

    def test_create_user_duplicate_telegram_id(self, repo: Repository):
        repo.create_user(telegram_id=100, name="Ana")
        with pytest.raises(sqlite3.IntegrityError):
            repo.create_user(telegram_id=100, name="Otra Ana")


class TestGetUser:
    def test_get_existing_user(self, repo: Repository, sample_user: User):
        found = repo.get_user_by_telegram_id(12345)
        assert found is not None
        assert found.name == "Juan Pérez"

    def test_get_nonexistent_user(self, repo: Repository):
        found = repo.get_user_by_telegram_id(99999)
        assert found is None

    def test_get_user_by_id(self, repo: Repository, sample_user: User):
        found = repo.get_user_by_id(sample_user.id)
        assert found is not None
        assert found.telegram_id == 12345


class TestCreateTicket:
    def test_create_ticket(self, repo: Repository, sample_user: User):
        ticket = repo.create_ticket(
            user_id=sample_user.id,
            image_path="/img/ticket.jpg",
            empresa="Repsol",
            nif="B12345678",
            descripcion="Gasolina 95",
            total=36.89,
            fecha="15/08/2026",
        )
        assert ticket.id is not None
        assert ticket.empresa == "Repsol"
        assert ticket.nif == "B12345678"
        assert ticket.total == 36.89

    def test_create_ticket_without_nif(self, repo: Repository, sample_user: User):
        ticket = repo.create_ticket(
            user_id=sample_user.id,
            image_path="/img/t.jpg",
            empresa="Bar",
            nif="",
            descripcion="Café",
            total=1.5,
            fecha="01/01/2026",
        )
        assert ticket.nif == ""


class TestGetTickets:
    def test_get_tickets_empty(self, repo: Repository, sample_user: User):
        tickets = repo.get_tickets_by_user(sample_user.id)
        assert tickets == []

    def test_get_tickets_only_for_user(self, repo: Repository):
        user1 = repo.create_user(telegram_id=1, name="User1")
        user2 = repo.create_user(telegram_id=2, name="User2")
        repo.create_ticket(
            user_id=user1.id, image_path="/a.jpg",
            empresa="A", nif="A1", descripcion="d",
            total=1.0, fecha="01/01/2026",
        )
        repo.create_ticket(
            user_id=user2.id, image_path="/b.jpg",
            empresa="B", nif="B2", descripcion="d",
            total=1.0, fecha="01/01/2026",
        )
        tickets_user1 = repo.get_tickets_by_user(user1.id)
        assert len(tickets_user1) == 1
        assert tickets_user1[0].empresa == "A"
        assert tickets_user1[0].nif == "A1"


class TestCountTickets:
    def test_count_zero(self, repo: Repository, sample_user: User):
        assert repo.count_tickets_by_user(sample_user.id) == 0

    def test_count_multiple(self, repo: Repository, sample_user: User):
        for i in range(5):
            repo.create_ticket(
                user_id=sample_user.id, image_path=f"/img/{i}.jpg",
                empresa="E", nif="", descripcion="d",
                total=1.0, fecha="01/01/2026",
            )
        assert repo.count_tickets_by_user(sample_user.id) == 5


class TestSyncStatusAndUpdates:
    def test_unsynced_and_mark_synced(self, repo: Repository, sample_user: User):
        t1 = repo.create_ticket(
            user_id=sample_user.id, image_path="/1.jpg",
            empresa="A", nif="", descripcion="d", total=10.0, fecha="01/01/2026",
            synced_to_excel=False,
        )
        t2 = repo.create_ticket(
            user_id=sample_user.id, image_path="/2.jpg",
            empresa="B", nif="", descripcion="d", total=20.0, fecha="01/01/2026",
            synced_to_excel=True,
        )

        unsynced = repo.get_unsynced_tickets()
        assert len(unsynced) == 1
        assert unsynced[0][0].id == t1.id
        assert unsynced[0][1] == sample_user.name

        repo.mark_tickets_synced([t1.id])
        assert len(repo.get_unsynced_tickets()) == 0

    def test_update_ticket(self, repo: Repository, sample_user: User):
        t = repo.create_ticket(
            user_id=sample_user.id, image_path="/1.jpg",
            empresa="Original", nif="", descripcion="d", total=100.0, fecha="01/01/2026",
        )
        repo.update_ticket(t.id, total=24.0, fecha="21/05/2024", empresa="Updated")
        tickets = repo.get_tickets_by_user(sample_user.id)
        assert tickets[0].total == 24.0
        assert tickets[0].fecha == "21/05/2024"
        assert tickets[0].empresa == "Updated"


class TestRolesAndAccountingRepo:
    def test_roles_and_dummy_workers(self, repo: Repository):
        admin = repo.create_user(telegram_id=1, name="Jefa", role="admin")
        assert admin.role == "admin"
        assert len(repo.get_admins()) == 1

        w1 = repo.create_dummy_worker("Trabajador Prueba 1")
        w2 = repo.create_dummy_worker("Trabajador Prueba 2")
        assert w1.telegram_id < 0
        assert w2.telegram_id < w1.telegram_id  # Unique decremented IDs
        assert w1.role == "trabajador"

        repo.set_user_role(w1.id, "admin")
        assert len(repo.get_admins()) == 2

    def test_cash_movements_and_balances(self, repo: Repository):
        admin = repo.create_user(telegram_id=1, name="Jefa", role="admin")
        worker = repo.create_user(telegram_id=2, name="Pedro", role="trabajador")

        # Case 1: Initial state -> Paz y Salvo
        b0 = repo.get_user_balance(worker.id)
        assert b0.net_balance == 0.0
        assert "PAZ Y SALVO" in b0.status_label

        # Case 2: Jefa assigns 50€ -> Worker owes 50€
        repo.create_cash_movement(
            user_id=worker.id, created_by_id=admin.id,
            movement_type="ASIGNACION", amount=50.0, note="Gasolina",
        )
        b1 = repo.get_user_balance(worker.id)
        assert b1.total_assigned == 50.0
        assert b1.net_balance == 50.0
        assert "DEBE A LA JEFA" in b1.status_label

        # Case 3: Worker spends 70€ in tickets -> Jefa owes worker 20€
        repo.create_ticket(
            user_id=worker.id, image_path="/t.jpg", empresa="Repsol",
            nif="", descripcion="Gasolina", total=70.0, fecha="01/01/2026",
        )
        b2 = repo.get_user_balance(worker.id)
        assert b2.total_spent == 70.0
        assert b2.net_balance == -20.0  # 50 - 70 = -20
        assert "LA JEFA LE DEBE" in b2.status_label
        assert "20.00 €" in b2.status_label

        # Case 4: Invalid movement type raises ValueError
        with pytest.raises(ValueError, match="Invalid movement_type"):
            repo.create_cash_movement(
                user_id=worker.id, created_by_id=admin.id,
                movement_type="INVALIDO", amount=10.0,
            )

