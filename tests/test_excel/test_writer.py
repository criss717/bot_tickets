"""Tests for Excel writer and synchronizer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from openpyxl import load_workbook

from db.models import Ticket
from db.repository import Repository
from excel.writer import append_ticket, ensure_workbook, rebuild_excel_from_db, sync_pending_tickets


@pytest.fixture
def excel_path(tmp_path: Path) -> Path:
    return tmp_path / "test_tickets.xlsx"


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "test.db")


@pytest.fixture
def sample_ticket() -> Ticket:
    return Ticket(
        user_id=1,
        image_path="/img/ticket.jpg",
        empresa="Repsol",
        nif="B12345678",
        descripcion="Gasolina 95",
        total=36.89,
        fecha="15/08/2026",
    )


class TestEnsureWorkbook:
    def test_creates_new_workbook(self, excel_path: Path):
        ensure_workbook(excel_path)
        assert excel_path.exists()

    def test_workbook_has_headers(self, excel_path: Path):
        ensure_workbook(excel_path)
        wb = load_workbook(excel_path)
        ws = wb.active
        headers = [ws.cell(row=1, column=i).value for i in range(1, 7)]
        assert headers == [
            "Trabajador", "Empresa", "NIF/CIF",
            "Descripción", "Total", "Fecha",
        ]

    def test_idempotent(self, excel_path: Path):
        ensure_workbook(excel_path)
        ensure_workbook(excel_path)
        wb = load_workbook(excel_path)
        ws = wb.active
        assert ws.max_row == 1


class TestAppendTicket:
    def test_append_first_ticket(self, excel_path: Path, sample_ticket: Ticket):
        row = append_ticket(excel_path, "Juan Pérez", sample_ticket)
        assert row == 2

    def test_append_writes_correct_data(self, excel_path: Path, sample_ticket: Ticket):
        append_ticket(excel_path, "Juan Pérez", sample_ticket)
        wb = load_workbook(excel_path)
        ws = wb.active
        assert ws.cell(row=2, column=1).value == "Juan Pérez"
        assert ws.cell(row=2, column=2).value == "Repsol"
        assert ws.cell(row=2, column=3).value == "B12345678"
        assert ws.cell(row=2, column=4).value == "Gasolina 95"
        assert ws.cell(row=2, column=5).value == 36.89
        assert ws.cell(row=2, column=6).value == "15/08/2026"

    def test_append_multiple_tickets(self, excel_path: Path):
        for i in range(3):
            ticket = Ticket(
                user_id=1, image_path=f"/img/{i}.jpg",
                empresa=f"Empresa {i}", nif=f"NIF{i}",
                descripcion=f"Desc {i}",
                total=10.0, fecha=f"0{i+1}/08/2026",
            )
            row = append_ticket(excel_path, "Ana", ticket)
            assert row == i + 2

        wb = load_workbook(excel_path)
        ws = wb.active
        assert ws.max_row == 4


class TestSyncPendingTickets:
    def test_sync_no_pending(self, excel_path: Path, repo: Repository):
        synced, ok = sync_pending_tickets(excel_path, repo)
        assert synced == 0
        assert ok is True

    def test_sync_pending_success(self, excel_path: Path, repo: Repository):
        user = repo.create_user(telegram_id=1, name="Carlos")
        t = repo.create_ticket(
            user_id=user.id, image_path="/img.jpg", empresa="Bar",
            nif="B123", descripcion="Café", total=2.5, fecha="01/01/2026",
            synced_to_excel=False,
        )

        synced, ok = sync_pending_tickets(excel_path, repo)
        assert synced == 1
        assert ok is True

        # Check Excel content
        wb = load_workbook(excel_path)
        ws = wb.active
        assert ws.max_row == 2
        assert ws.cell(row=2, column=1).value == "Carlos"
        assert ws.cell(row=2, column=5).value == 2.5

        # Check DB status
        assert len(repo.get_unsynced_tickets()) == 0

    def test_sync_handles_permission_error_gracefully(self, excel_path: Path, repo: Repository):
        user = repo.create_user(telegram_id=1, name="Carlos")
        t = repo.create_ticket(
            user_id=user.id, image_path="/img.jpg", empresa="Bar",
            nif="B123", descripcion="Café", total=2.5, fecha="01/01/2026",
            synced_to_excel=False,
        )

        with patch("openpyxl.Workbook.save", side_effect=PermissionError("Locked by Excel")):
            synced, ok = sync_pending_tickets(excel_path, repo)

        # Must return (0, False) without raising exception
        assert synced == 0
        assert ok is False

        # Ticket must still be pending in DB
        unsynced = repo.get_unsynced_tickets()
        assert len(unsynced) == 1
        assert unsynced[0][0].id == t.id


class TestRebuildExcelFromDb:
    def test_rebuild_success(self, excel_path: Path, repo: Repository):
        user = repo.create_user(telegram_id=1, name="Carlos")
        for i in range(3):
            repo.create_ticket(
                user_id=user.id, image_path=f"/{i}.jpg", empresa=f"E{i}",
                nif=f"N{i}", descripcion=f"D{i}", total=float(i + 1), fecha="01/01/2026",
                synced_to_excel=False,
            )

        synced, ok = rebuild_excel_from_db(excel_path, repo)
        assert synced == 3
        assert ok is True

        wb = load_workbook(excel_path)
        ws = wb.active
        assert ws.max_row == 4
        assert len(repo.get_unsynced_tickets()) == 0


class TestMultiSheetAccounting:
    def test_sync_movements_and_resumen_calculation(self, excel_path: Path, repo: Repository):
        admin = repo.create_user(telegram_id=999, name="La Jefa", role="admin")
        worker = repo.create_user(telegram_id=101, name="Juan Chofer", role="trabajador")

        # 1. Admin assigns 100€ to worker
        mov1 = repo.create_cash_movement(
            user_id=worker.id,
            created_by_id=admin.id,
            movement_type="ASIGNACION",
            amount=100.0,
            note="Fondo inicial semana",
            fecha="10/11/2026",
        )

        # 2. Worker spends 60€ in a real ticket
        t1 = repo.create_ticket(
            user_id=worker.id,
            image_path="/t1.jpg",
            empresa="Gasolinera Repsol",
            nif="A12345678",
            descripcion="Diésel",
            total=60.0,
            fecha="11/11/2026",
            synced_to_excel=False,
        )

        # 3. Worker returns 40€ to Jefa
        mov2 = repo.create_cash_movement(
            user_id=worker.id,
            created_by_id=worker.id,
            movement_type="DEVOLUCION",
            amount=40.0,
            note="Devolución sobrante viernes",
            fecha="12/11/2026",
        )

        # Sync to Excel
        synced_count, ok = sync_pending_tickets(excel_path, repo)
        assert ok is True
        assert synced_count == 3  # 1 ticket + 2 movements

        wb = load_workbook(excel_path)
        assert set(wb.sheetnames) == {"Resumen Cuentas", "Resumen Mensual", "Tickets", "Movimientos Caja"}

        # Verify Movimientos Caja sheet
        ws_caja = wb["Movimientos Caja"]
        assert ws_caja.max_row == 3
        assert ws_caja.cell(row=2, column=1).value == "Juan Chofer"
        assert ws_caja.cell(row=2, column=2).value == "ASIGNACION"
        assert ws_caja.cell(row=2, column=3).value == 100.0
        assert ws_caja.cell(row=2, column=6).value == "La Jefa"

        assert ws_caja.cell(row=3, column=1).value == "Juan Chofer"
        assert ws_caja.cell(row=3, column=2).value == "DEVOLUCION"
        assert ws_caja.cell(row=3, column=3).value == 40.0
        assert ws_caja.cell(row=3, column=6).value == "Juan Chofer"

        # Verify Resumen Cuentas sheet: Juan Chofer should be A PAZ Y SALVO (100 - 60 - 40 = 0)
        ws_resumen = wb["Resumen Cuentas"]
        # Row 2 is La Jefa, Row 3 is Juan Chofer (or sorted by ID)
        worker_row = None
        for r in range(2, ws_resumen.max_row + 1):
            if ws_resumen.cell(row=r, column=1).value == "Juan Chofer":
                worker_row = r
                break
        assert worker_row is not None
        assert ws_resumen.cell(row=worker_row, column=3).value == 100.0  # Asignado
        assert ws_resumen.cell(row=worker_row, column=4).value == 60.0   # Gastado
        assert ws_resumen.cell(row=worker_row, column=5).value == 40.0   # Devuelto
        assert ws_resumen.cell(row=worker_row, column=7).value == 0.0    # Saldo Neto
        assert "PAZ Y SALVO" in ws_resumen.cell(row=worker_row, column=8).value

    def test_status_colors_and_formats(self, excel_path: Path, repo: Repository):
        admin = repo.create_user(telegram_id=1, name="Jefa", role="admin")
        w_debe = repo.create_user(telegram_id=2, name="Deudor", role="trabajador")
        w_acreedor = repo.create_user(telegram_id=3, name="Acreedor", role="trabajador")

        # Deudor has 50€ assigned, spends 0€ -> Debe a la jefa
        repo.create_cash_movement(
            user_id=w_debe.id, created_by_id=admin.id,
            movement_type="ASIGNACION", amount=50.0, note="",
        )

        # Acreedor has 0€ assigned, spends 80€ ticket -> La jefa le debe
        repo.create_ticket(
            user_id=w_acreedor.id, image_path="/t.jpg",
            empresa="Taller", nif="B999", descripcion="Reparación",
            total=80.0, fecha="01/01/2026", synced_to_excel=False,
        )

        rebuild_excel_from_db(excel_path, repo)

        wb = load_workbook(excel_path)
        ws_resumen = wb["Resumen Cuentas"]

        row_debe = None
        row_acreedor = None
        for r in range(2, ws_resumen.max_row + 1):
            name = ws_resumen.cell(row=r, column=1).value
            if name == "Deudor":
                row_debe = r
            elif name == "Acreedor":
                row_acreedor = r

        assert row_debe is not None
        assert row_acreedor is not None

        # Verify Debe cell styles (yellow)
        cell_debe = ws_resumen.cell(row=row_debe, column=8)
        assert "DEBE A LA EMPRESA" in str(cell_debe.value)
        assert cell_debe.fill.start_color.rgb == "00FFEB9C" or cell_debe.fill.fill_type == "solid"

        # Verify Acreedor cell styles (red)
        cell_acreedor = ws_resumen.cell(row=row_acreedor, column=8)
        assert "LA EMPRESA LE DEBE" in str(cell_acreedor.value)
        assert cell_acreedor.fill.start_color.rgb == "00FFC7CE" or cell_acreedor.fill.fill_type == "solid"

        # Verify number formats
        assert ws_resumen.cell(row=row_debe, column=3).number_format == "#,##0.00 €"
        assert ws_resumen.cell(row=row_debe, column=7).number_format == "#,##0.00 €"

    def test_desasignacion_updates_resumen_and_cleans_temp_files(self, excel_path: Path, repo: Repository):
        admin = repo.create_user(telegram_id=1, name="Jefa", role="admin")
        worker = repo.create_user(telegram_id=2, name="Pepe", role="trabajador")

        repo.create_cash_movement(user_id=worker.id, created_by_id=admin.id, movement_type="ASIGNACION", amount=100.0)
        repo.create_cash_movement(user_id=worker.id, created_by_id=admin.id, movement_type="DESASIGNACION", amount=100.0, note="Equivocación")

        synced, ok = sync_pending_tickets(excel_path, repo)
        assert ok is True and synced == 2

        wb = load_workbook(excel_path)
        ws_resumen = wb["Resumen Cuentas"]
        for r in range(2, ws_resumen.max_row + 1):
            if ws_resumen.cell(row=r, column=1).value == "Pepe":
                assert ws_resumen.cell(row=r, column=3).value == 0.0
                assert ws_resumen.cell(row=r, column=7).value == 0.0
                assert "PAZ Y SALVO" in ws_resumen.cell(row=r, column=8).value
                break
        # Ensure atomic save left zero leftover temp files
        assert [p for p in excel_path.parent.glob(".tmp_*")] == []

    def test_sync_auto_rebuilds_when_db_migrated_with_synced_flags_and_missing_excel(
        self, excel_path: Path, repo: Repository
    ):
        worker = repo.create_user(telegram_id=101, name="Pepe", role="trabajador")
        repo.create_ticket(
            user_id=worker.id,
            image_path="/img/old1.jpg",
            empresa="Restaurante Viejo SL",
            nif="B11111111",
            descripcion="Comida",
            total=25.0,
            fecha="10/11/2026",
            synced_to_excel=True,
        )
        repo.create_ticket(
            user_id=worker.id,
            image_path="/img/old2.jpg",
            empresa="Gasolinera Antigua",
            nif="",
            descripcion="Gasoil",
            total=45.0,
            fecha="11/11/2026",
            synced_to_excel=True,
        )

        # Simulate brand-new server where export/tickets.xlsx does not exist yet
        assert not excel_path.exists()
        synced, ok = sync_pending_tickets(excel_path, repo)
        assert ok is True
        assert synced == 2

        wb = load_workbook(excel_path)
        assert wb["Tickets"].max_row == 3  # Header + 2 historical tickets

    def test_sync_auto_rebuilds_when_excel_missing_historical_rows(
        self, tmp_path: Path, repo: Repository
    ):
        export_dir = tmp_path / "export"
        excel_path = export_dir / "tickets.xlsx"
        photo_path = export_dir / "fotos" / "2026-11" / "2026-11-16_Pepe_Tienda_30-00EUR_abc123.jpg"
        photo_path.parent.mkdir(parents=True, exist_ok=True)
        photo_path.write_bytes(b"img")

        admin = repo.create_user(telegram_id=999, name="Admin", role="admin")
        worker = repo.create_user(telegram_id=101, name="Pepe", role="trabajador")
        # 2 historical tickets already marked synced=1 on old PC
        repo.create_ticket(
            user_id=worker.id, image_path=str(photo_path), empresa="Tienda",
            nif="", descripcion="Material", total=30.0, fecha="16/11/2026", synced_to_excel=True,
        )
        repo.create_ticket(
            user_id=worker.id, image_path="", empresa="Peaje",
            nif="", descripcion="Autopista", total=10.0, fecha="15/11/2026", synced_to_excel=True,
        )
        m1 = repo.create_cash_movement(
            user_id=worker.id, created_by_id=admin.id,
            movement_type="ASIGNACION", amount=100.0, note="Fondo",
        )
        repo.mark_movements_synced([m1.id])

        # Create an empty Excel workbook (simulating server startup ensure_workbook before fix)
        ensure_workbook(excel_path)
        assert load_workbook(excel_path)["Tickets"].max_row == 1

        synced, ok = sync_pending_tickets(excel_path, repo)
        assert ok is True
        assert synced == 3  # 2 tickets + 1 movement recovered

        wb = load_workbook(excel_path)
        ws_tickets = wb["Tickets"]
        assert ws_tickets.max_row == 3
        assert wb["Movimientos Caja"].max_row == 2
        # Verify column 8 (Foto) contains relative link inside export/
        foto_cell = ws_tickets.cell(row=2, column=8)
        assert foto_cell.value == "fotos/2026-11/2026-11-16_Pepe_Tienda_30-00EUR_abc123.jpg"
        assert foto_cell.hyperlink is not None
        assert foto_cell.hyperlink.target == "fotos/2026-11/2026-11-16_Pepe_Tienda_30-00EUR_abc123.jpg"


