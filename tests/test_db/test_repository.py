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
        assert "DEBE A LA EMPRESA" in b1.status_label

        # Case 3: Worker spends 70€ in tickets -> Empresa owes worker 20€
        repo.create_ticket(
            user_id=worker.id, image_path="/t.jpg", empresa="Repsol",
            nif="", descripcion="Gasolina", total=70.0, fecha="01/01/2026",
        )
        b2 = repo.get_user_balance(worker.id)
        assert b2.total_spent == 70.0
        assert b2.net_balance == -20.0  # 50 - 70 = -20
        assert "LA EMPRESA LE DEBE" in b2.status_label
        assert "20.00 €" in b2.status_label

        # Case 4: Invalid movement type raises ValueError
        with pytest.raises(ValueError, match="Invalid movement_type"):
            repo.create_cash_movement(
                user_id=worker.id, created_by_id=admin.id,
                movement_type="INVALIDO", amount=10.0,
            )


class TestUnassignmentDeletionAndDeduplication:
    def test_desasignacion_reduces_assigned_and_net_balance(self, repo: Repository):
        admin = repo.create_user(telegram_id=1, name="Jefa", role="admin")
        worker = repo.create_user(telegram_id=2, name="Lucas", role="trabajador")

        repo.create_cash_movement(user_id=worker.id, created_by_id=admin.id, movement_type="ASIGNACION", amount=100.0)
        repo.create_cash_movement(user_id=worker.id, created_by_id=admin.id, movement_type="DESASIGNACION", amount=40.0, note="Error")

        bal = repo.get_user_balance(worker.id)
        assert bal.total_assigned == 60.0
        assert bal.month_assigned == 60.0
        assert bal.net_balance == 60.0

    def test_delete_cash_movement_and_ticket(self, repo: Repository):
        admin = repo.create_user(telegram_id=1, name="Jefa", role="admin")
        worker = repo.create_user(telegram_id=2, name="Mario", role="trabajador")
        mov = repo.create_cash_movement(user_id=worker.id, created_by_id=admin.id, movement_type="ASIGNACION", amount=500.0)
        tkt = repo.create_ticket(
            user_id=worker.id, image_path="/img/x.jpg",
            empresa="Bar", nif="", descripcion="Café", total=2.5, fecha="13/11/2026",
        )

        deleted_mov = repo.delete_cash_movement(mov.id)
        assert deleted_mov is not None and deleted_mov.amount == 500.0
        assert repo.get_movement_by_id(mov.id) is None

        deleted_tkt = repo.delete_ticket(tkt.id)
        assert deleted_tkt is not None and deleted_tkt.id == tkt.id
        assert repo.get_ticket_by_id(tkt.id) is None
        assert repo.get_user_balance(worker.id).net_balance == 0.0

    def test_update_name_and_delete_dummy_worker(self, repo: Repository):
        dummy_empty = repo.create_dummy_worker("Test Borrar")
        dummy_busy = repo.create_dummy_worker("Con Datos")
        repo.create_ticket(
            user_id=dummy_busy.id, image_path="/img/b.jpg",
            empresa="Tienda", nif="", descripcion="Algo", total=5.0, fecha="13/11/2026",
        )

        repo.update_user_name(dummy_busy.id, "Nombre Oficial")
        assert repo.get_user_by_id(dummy_busy.id).name == "Nombre Oficial"

        assert repo.delete_dummy_worker(dummy_empty.id) is True
        assert repo.delete_dummy_worker(dummy_busy.id) is False  # Has ticket

    def test_global_duplicate_detection_respects_date_window(self, repo: Repository):
        w1 = repo.create_user(telegram_id=10, name="Carlos")
        repo.create_user(telegram_id=20, name="Ana")

        # Recent ticket uploaded by w1
        recent = repo.create_ticket(
            user_id=w1.id,
            image_path="/img/1.jpg",
            empresa="Restaurante Montemayor SL",
            nif="B90321852",
            descripcion="Menú",
            total=28.40,
            fecha="13/11/2026",
            image_sha256="abc123sha",
            image_dhash="0f0f0f0f0f0f0f0f",
        )
        # Old ticket outside 90-day window
        old = repo.create_ticket(
            user_id=w1.id,
            image_path="/img/old.jpg",
            empresa="Antiguo Bar SL",
            nif="",
            descripcion="Café",
            total=15.0,
            fecha="01/01/2025",
            image_sha256="oldsha",
            image_dhash="aaaaaaaaaaaaaaaa",
        )
        with repo._connect() as conn:
            conn.execute("UPDATE tickets SET created_at = '2025-01-01 10:00:00' WHERE id = ?", (old.id,))

        # 1. Exact SHA-256 hit across whole company
        hit_sha = repo.find_duplicate_by_image_hash("abc123sha", "")
        assert hit_sha is not None
        assert hit_sha[0].id == recent.id
        assert hit_sha[1] == "Carlos"
        assert hit_sha[2] == "exact"

        # 2. Visual dHash (1 bit difference) hit inside 90-day window
        hit_dhash = repo.find_duplicate_by_image_hash("different_sha", "0f0f0f0f0f0f0f0e", max_hamming=4, days_window=90)
        assert hit_dhash is not None
        assert hit_dhash[0].id == recent.id
        assert hit_dhash[2] == "visual"

        # 3. Old dHash outside 90-day window must NOT trigger scan/match
        assert repo.find_duplicate_by_image_hash("other", "aaaaaaaaaaaaaaab", max_hamming=4, days_window=90) is None

        # 4. Semantic match via indexed (fecha, total) + fuzzy company
        hit_data = repo.find_duplicate_by_data("Montemayor", 28.40, "13/11/2026", days_window=90)
        assert hit_data is not None and hit_data[0].id == recent.id

        # 5. Old data ticket outside 90-day window ignored
        assert repo.find_duplicate_by_data("Antiguo Bar SL", 15.0, "01/01/2025", days_window=90) is None

    def test_sync_admin_roles_promotes_demotes_and_preserves_bot_admins(self, repo: Repository):
        ex_env_admin = repo.create_user(telegram_id=101, name="Ex Admin Env", role="admin", admin_source="env")
        new_env_admin = repo.create_user(telegram_id=202, name="Nuevo Admin Env", role="trabajador")
        bot_admin = repo.create_user(telegram_id=404, name="Lucia Bot Admin", role="trabajador")
        repo.set_user_role(bot_admin.id, "admin", admin_source="bot")
        worker = repo.create_user(telegram_id=303, name="Currito", role="trabajador")
        dummy = repo.create_dummy_worker("Sin Telegram")

        # Promote 202 from .env, demote 101 removed from .env, and KEEP 404 promoted via bot
        promoted, demoted = repo.sync_admin_roles((202,))
        assert promoted == 1
        assert demoted == 1
        assert repo.get_user_by_id(ex_env_admin.id).role == "trabajador"
        assert repo.get_user_by_id(new_env_admin.id).role == "admin"
        assert repo.get_user_by_id(new_env_admin.id).admin_source == "env"
        assert repo.get_user_by_id(bot_admin.id).role == "admin"
        assert repo.get_user_by_id(bot_admin.id).admin_source == "bot"
        assert repo.get_user_by_id(worker.id).role == "trabajador"
        assert repo.get_user_by_id(dummy.id).role == "trabajador"
        assert {u.telegram_id for u in repo.get_admins()} == {202, 404}

        # Removing 202 from .env demotes 202, while keeping bot-promoted 404
        promoted2, demoted2 = repo.sync_admin_roles(())
        assert promoted2 == 0
        assert demoted2 == 1
        assert repo.get_user_by_id(new_env_admin.id).role == "trabajador"
        assert repo.get_user_by_id(bot_admin.id).role == "admin"
        assert [u.telegram_id for u in repo.get_admins()] == [404]

