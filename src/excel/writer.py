"""Excel writer and synchronizer for tickets, cash movements, and worker balances."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

from db.models import CashMovement, MonthlyWorkerSummary, Ticket, UserBalance

if TYPE_CHECKING:
    from db.repository import Repository

logger = logging.getLogger(__name__)


# ── Column definitions & formatting ───────────────────────────────────────

COLUMNS_RESUMEN = [
    ("Trabajador", 22),
    ("Rol", 14),
    ("Entregado (Mes)", 18),
    ("Gastado (Mes)", 20),
    ("Devuelto (Mes)", 18),
    ("Saldo Mes Ant.", 18),
    ("SALDO NETO ACTUAL", 20),
    ("ESTADO CUENTA", 24),
]

COLUMNS_MENSUAL = [
    ("Mes (Año-Mes)", 16),
    ("Trabajador", 22),
    ("Rol", 14),
    ("Nº Tickets", 12),
    ("Total Entregado", 18),
    ("Total Gastado", 18),
    ("Total Devuelto", 18),
    ("Balance del Mes", 20),
]

COLUMNS_TICKETS = [
    ("Trabajador", 20),
    ("Empresa", 30),
    ("NIF/CIF", 15),
    ("Descripción", 35),
    ("Total", 14),
    ("Fecha", 14),
    ("Mes Liquidación", 16),
]

COLUMNS_CAJA = [
    ("Trabajador", 20),
    ("Tipo Movimiento", 18),
    ("Importe", 14),
    ("Concepto / Nota", 35),
    ("Fecha", 14),
    ("Registrado Por", 20),
    ("Mes", 14),
]

HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
HEADER_FILL = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
HEADER_ALIGNMENT = Alignment(horizontal="center", vertical="center")

TOTAL_FONT = Font(bold=True, color="000000", size=11)
TOTAL_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

# Traffic-light styles for account status
FILL_PAZ = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
FONT_PAZ = Font(color="006100", bold=True)

FILL_DEBE = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
FONT_DEBE = Font(color="9C6500", bold=True)

FILL_ACREEDOR = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
FONT_ACREEDOR = Font(color="9C0006", bold=True)


def _format_created_month(dt_or_str: Any) -> str:
    """Safely extract YYYY-MM from a datetime object or timestamp string."""
    if not dt_or_str:
        return ""
    if hasattr(dt_or_str, "strftime"):
        return dt_or_str.strftime("%Y-%m")
    s = str(dt_or_str).strip()
    return s[:7] if len(s) >= 7 else s


def _setup_sheet_headers(ws, columns: list[tuple[str, int]]) -> None:
    """Apply styling, widths, and freeze header for a worksheet."""
    ws.freeze_panes = "A2"
    for col_idx, (header, width) in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = HEADER_ALIGNMENT
        col_letter = chr(64 + col_idx) if col_idx <= 26 else f"A{chr(64 + col_idx - 26)}"
        ws.column_dimensions[col_letter].width = width


def _save_workbook_atomic(wb: Workbook, excel_path: Path) -> None:
    """Save workbook atomically via same-directory temp file with 0644 permissions for Samba read-only sharing."""
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=excel_path.parent,
            prefix=".tmp_",
            suffix=".xlsx",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        wb.save(tmp_path)
        try:
            os.chmod(tmp_path, 0o644)
        except OSError:
            pass
        os.replace(tmp_path, excel_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def ensure_workbook(excel_path: Path) -> None:
    """Ensure the Excel file exists with all 4 required sheets."""
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    if not excel_path.exists():
        wb = Workbook()

        # Sheet 1: Resumen Cuentas
        ws_resumen = wb.active
        ws_resumen.title = "Resumen Cuentas"
        _setup_sheet_headers(ws_resumen, COLUMNS_RESUMEN)

        # Sheet 2: Resumen Mensual
        ws_mensual = wb.create_sheet(title="Resumen Mensual")
        _setup_sheet_headers(ws_mensual, COLUMNS_MENSUAL)

        # Sheet 3: Tickets
        ws_tickets = wb.create_sheet(title="Tickets")
        _setup_sheet_headers(ws_tickets, COLUMNS_TICKETS)

        # Sheet 4: Movimientos Caja
        ws_caja = wb.create_sheet(title="Movimientos Caja")
        _setup_sheet_headers(ws_caja, COLUMNS_CAJA)

        wb.active = ws_tickets
        _save_workbook_atomic(wb, excel_path)
        return

    # If it exists, ensure all 4 sheets are present
    wb = load_workbook(excel_path)
    dirty = False

    if "Resumen Cuentas" not in wb.sheetnames:
        ws_resumen = wb.create_sheet(title="Resumen Cuentas", index=0)
        _setup_sheet_headers(ws_resumen, COLUMNS_RESUMEN)
        dirty = True

    if "Resumen Mensual" not in wb.sheetnames:
        ws_mensual = wb.create_sheet(title="Resumen Mensual", index=1)
        _setup_sheet_headers(ws_mensual, COLUMNS_MENSUAL)
        dirty = True

    if "Tickets" not in wb.sheetnames:
        ws_tickets = wb.create_sheet(title="Tickets", index=2)
        _setup_sheet_headers(ws_tickets, COLUMNS_TICKETS)
        dirty = True

    if "Movimientos Caja" not in wb.sheetnames:
        ws_caja = wb.create_sheet(title="Movimientos Caja", index=3)
        _setup_sheet_headers(ws_caja, COLUMNS_CAJA)
        dirty = True

    if dirty:
        _save_workbook_atomic(wb, excel_path)


def append_ticket(excel_path: Path, user_name: str, ticket: Ticket) -> int:
    """Append a single ticket to the Tickets sheet.

    Raises PermissionError if the file is locked by Excel on Windows.
    """
    ensure_workbook(excel_path)
    wb = load_workbook(excel_path)
    ws = wb["Tickets"] if "Tickets" in wb.sheetnames else wb.active

    next_row = ws.max_row + 1
    month_liq = _format_created_month(ticket.created_at)

    ws.cell(row=next_row, column=1, value=user_name)
    ws.cell(row=next_row, column=2, value=ticket.empresa)
    ws.cell(row=next_row, column=3, value=ticket.nif)
    ws.cell(row=next_row, column=4, value=ticket.descripcion)
    ws.cell(row=next_row, column=5, value=float(ticket.total))
    ws.cell(row=next_row, column=6, value=ticket.fecha)
    ws.cell(row=next_row, column=7, value=month_liq)

    ws.cell(row=next_row, column=5).number_format = "#,##0.00 €"

    # Refresh auto-filter if data present
    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions

    _save_workbook_atomic(wb, excel_path)
    return next_row


def update_resumen_sheet(ws, balances: list[UserBalance]) -> None:
    """Rewrite rows of Resumen Cuentas sheet with fresh calculations and Totals."""
    ws.freeze_panes = "A2"
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)

    tot_assigned = 0.0
    tot_spent = 0.0
    tot_returned = 0.0
    tot_prev = 0.0
    tot_net = 0.0

    last_row = 1
    for idx, bal in enumerate(balances, start=2):
        last_row = idx
        ws.cell(row=idx, column=1, value=bal.name)
        ws.cell(row=idx, column=2, value=bal.role.capitalize())

        # Monthly Activity
        c_asig = ws.cell(row=idx, column=3, value=bal.month_assigned)
        c_asig.number_format = "#,##0.00 €"
        tot_assigned += bal.month_assigned

        c_spent = ws.cell(row=idx, column=4, value=bal.month_spent)
        c_spent.number_format = "#,##0.00 €"
        tot_spent += bal.month_spent

        c_ret = ws.cell(row=idx, column=5, value=bal.month_returned)
        c_ret.number_format = "#,##0.00 €"
        tot_returned += bal.month_returned

        # Previous carry-over
        c_prev = ws.cell(row=idx, column=6, value=bal.previous_balance)
        c_prev.number_format = "#,##0.00 €"
        tot_prev += bal.previous_balance

        # Living Net Balance
        c_net = ws.cell(row=idx, column=7, value=bal.net_balance)
        c_net.number_format = "#,##0.00 €"
        tot_net += bal.net_balance

        # Status Label & Traffic Light
        c_status = ws.cell(row=idx, column=8, value=bal.status_label)
        c_status.alignment = Alignment(horizontal="center")

        if "PAZ" in bal.status_label:
            c_status.fill = FILL_PAZ
            c_status.font = FONT_PAZ
        elif "DEBE A LA JEFA" in bal.status_label:
            c_status.fill = FILL_DEBE
            c_status.font = FONT_DEBE
        else:
            c_status.fill = FILL_ACREEDOR
            c_status.font = FONT_ACREEDOR

    # Total row for company overview
    if balances:
        r_tot = last_row + 1
        ws.cell(row=r_tot, column=1, value="TOTALES EMPRESA").font = TOTAL_FONT
        ws.cell(row=r_tot, column=2, value="").fill = TOTAL_FILL

        for c_idx, val in (
            (3, tot_assigned),
            (4, tot_spent),
            (5, tot_returned),
            (6, tot_prev),
            (7, tot_net),
        ):
            c = ws.cell(row=r_tot, column=c_idx, value=round(val, 2))
            c.font = TOTAL_FONT
            c.fill = TOTAL_FILL
            c.number_format = "#,##0.00 €"

        c_stat_tot = ws.cell(row=r_tot, column=8, value="")
        c_stat_tot.fill = TOTAL_FILL

    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions


def update_mensual_sheet(ws, summaries: list[MonthlyWorkerSummary]) -> None:
    """Rewrite rows of Resumen Mensual with historical month-by-month closures."""
    ws.freeze_panes = "A2"
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)

    for idx, s in enumerate(summaries, start=2):
        ws.cell(row=idx, column=1, value=s.year_month)
        ws.cell(row=idx, column=2, value=s.user_name)
        ws.cell(row=idx, column=3, value=s.role.capitalize())
        ws.cell(row=idx, column=4, value=s.tickets_count)

        c_asig = ws.cell(row=idx, column=5, value=s.total_assigned)
        c_asig.number_format = "#,##0.00 €"

        c_spent = ws.cell(row=idx, column=6, value=s.total_spent)
        c_spent.number_format = "#,##0.00 €"

        c_ret = ws.cell(row=idx, column=7, value=s.total_returned)
        c_ret.number_format = "#,##0.00 €"

        c_net = ws.cell(row=idx, column=8, value=s.net_month)
        c_net.number_format = "#,##0.00 €"

    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions


def sync_pending_tickets(excel_path: Path, repo: Repository) -> tuple[int, bool]:
    """Synchronize pending tickets, pending cash movements, and refresh summary sheets.

    Args:
        excel_path: Path to the Excel file.
        repo: Repository containing ticket and accounting data.

    Returns:
        (synced_items_count, success_flag)
    """
    pending_tickets = repo.get_unsynced_tickets()
    pending_movements = repo.get_unsynced_movements()

    if not pending_tickets and not pending_movements:
        return 0, True

    try:
        ensure_workbook(excel_path)
        wb = load_workbook(excel_path)

        # 1. Sync pending tickets to Tickets sheet
        ws_tickets = wb["Tickets"]
        synced_ticket_ids: list[int] = []
        for ticket, user_name in pending_tickets:
            next_row = ws_tickets.max_row + 1
            month_liq = _format_created_month(ticket.created_at)

            ws_tickets.cell(row=next_row, column=1, value=user_name)
            ws_tickets.cell(row=next_row, column=2, value=ticket.empresa)
            ws_tickets.cell(row=next_row, column=3, value=ticket.nif)
            ws_tickets.cell(row=next_row, column=4, value=ticket.descripcion)
            ws_tickets.cell(row=next_row, column=5, value=float(ticket.total))
            ws_tickets.cell(row=next_row, column=6, value=ticket.fecha)
            ws_tickets.cell(row=next_row, column=7, value=month_liq)
            ws_tickets.cell(row=next_row, column=5).number_format = "#,##0.00 €"
            if ticket.id is not None:
                synced_ticket_ids.append(ticket.id)

        if ws_tickets.max_row > 1:
            ws_tickets.auto_filter.ref = ws_tickets.dimensions

        # 2. Sync pending movements to Movimientos Caja sheet
        ws_caja = wb["Movimientos Caja"]
        synced_movement_ids: list[int] = []
        for mov, worker_name, creator_name in pending_movements:
            next_row = ws_caja.max_row + 1
            month_mov = _format_created_month(mov.created_at)

            ws_caja.cell(row=next_row, column=1, value=worker_name)
            ws_caja.cell(row=next_row, column=2, value=mov.movement_type)
            ws_caja.cell(row=next_row, column=3, value=float(mov.amount))
            ws_caja.cell(row=next_row, column=4, value=mov.note)
            ws_caja.cell(row=next_row, column=5, value=mov.fecha)
            ws_caja.cell(row=next_row, column=6, value=creator_name)
            ws_caja.cell(row=next_row, column=7, value=month_mov)
            ws_caja.cell(row=next_row, column=3).number_format = "#,##0.00 €"
            if mov.id is not None:
                synced_movement_ids.append(mov.id)

        if ws_caja.max_row > 1:
            ws_caja.auto_filter.ref = ws_caja.dimensions

        # 3. Refresh Resumen Cuentas sheet
        ws_resumen = wb["Resumen Cuentas"]
        balances = repo.get_all_balances()
        update_resumen_sheet(ws_resumen, balances)

        # 4. Refresh Resumen Mensual sheet
        ws_mensual = wb["Resumen Mensual"]
        summaries = repo.get_monthly_summaries()
        update_mensual_sheet(ws_mensual, summaries)

        wb.active = ws_tickets
        _save_workbook_atomic(wb, excel_path)

        if synced_ticket_ids:
            repo.mark_tickets_synced(synced_ticket_ids)
        if synced_movement_ids:
            repo.mark_movements_synced(synced_movement_ids)

        total_synced = len(synced_ticket_ids) + len(synced_movement_ids)
        logger.info(
            f"Successfully synchronized {len(synced_ticket_ids)} tickets "
            f"and {len(synced_movement_ids)} cash movements to Excel."
        )
        return total_synced, True

    except PermissionError:
        logger.warning(
            f"Excel file {excel_path} is currently locked (likely open in Excel). "
            f"{len(pending_tickets)} ticket(s) and {len(pending_movements)} movement(s) remain pending in SQLite."
        )
        return 0, False
    except Exception as e:
        logger.error(f"Unexpected error while synchronizing to Excel: {type(e).__name__}: {e}")
        return 0, False


def rebuild_excel_from_db(excel_path: Path, repo: Repository) -> tuple[int, bool]:
    """Completely rebuild the Excel file with all 4 sheets from DB data."""
    all_tickets = repo.get_all_tickets_with_users()
    all_movements = repo.get_all_movements_with_users()
    all_balances = repo.get_all_balances()
    all_summaries = repo.get_monthly_summaries()

    try:
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        wb = Workbook()

        # Sheet 1: Resumen Cuentas
        ws_resumen = wb.active
        ws_resumen.title = "Resumen Cuentas"
        _setup_sheet_headers(ws_resumen, COLUMNS_RESUMEN)
        update_resumen_sheet(ws_resumen, all_balances)

        # Sheet 2: Resumen Mensual
        ws_mensual = wb.create_sheet(title="Resumen Mensual")
        _setup_sheet_headers(ws_mensual, COLUMNS_MENSUAL)
        update_mensual_sheet(ws_mensual, all_summaries)

        # Sheet 3: Tickets
        ws_tickets = wb.create_sheet(title="Tickets")
        _setup_sheet_headers(ws_tickets, COLUMNS_TICKETS)
        synced_ticket_ids: list[int] = []
        for idx, (ticket, user_name) in enumerate(all_tickets, start=2):
            month_liq = _format_created_month(ticket.created_at)
            ws_tickets.cell(row=idx, column=1, value=user_name)
            ws_tickets.cell(row=idx, column=2, value=ticket.empresa)
            ws_tickets.cell(row=idx, column=3, value=ticket.nif)
            ws_tickets.cell(row=idx, column=4, value=ticket.descripcion)
            ws_tickets.cell(row=idx, column=5, value=float(ticket.total))
            ws_tickets.cell(row=idx, column=6, value=ticket.fecha)
            ws_tickets.cell(row=idx, column=7, value=month_liq)
            ws_tickets.cell(row=idx, column=5).number_format = "#,##0.00 €"
            if ticket.id is not None:
                synced_ticket_ids.append(ticket.id)

        if ws_tickets.max_row > 1:
            ws_tickets.auto_filter.ref = ws_tickets.dimensions

        # Sheet 4: Movimientos Caja
        ws_caja = wb.create_sheet(title="Movimientos Caja")
        _setup_sheet_headers(ws_caja, COLUMNS_CAJA)
        synced_movement_ids: list[int] = []
        for idx, (mov, worker_name, creator_name) in enumerate(all_movements, start=2):
            month_mov = _format_created_month(mov.created_at)
            ws_caja.cell(row=idx, column=1, value=worker_name)
            ws_caja.cell(row=idx, column=2, value=mov.movement_type)
            ws_caja.cell(row=idx, column=3, value=float(mov.amount))
            ws_caja.cell(row=idx, column=4, value=mov.note)
            ws_caja.cell(row=idx, column=5, value=mov.fecha)
            ws_caja.cell(row=idx, column=6, value=creator_name)
            ws_caja.cell(row=idx, column=7, value=month_mov)
            ws_caja.cell(row=idx, column=3).number_format = "#,##0.00 €"
            if mov.id is not None:
                synced_movement_ids.append(mov.id)

        if ws_caja.max_row > 1:
            ws_caja.auto_filter.ref = ws_caja.dimensions

        wb.active = ws_tickets
        _save_workbook_atomic(wb, excel_path)

        if synced_ticket_ids:
            repo.mark_tickets_synced(synced_ticket_ids)
        if synced_movement_ids:
            repo.mark_movements_synced(synced_movement_ids)

        logger.info(
            f"Rebuilt Excel with {len(synced_ticket_ids)} tickets, "
            f"{len(synced_movement_ids)} cash movements, and {len(all_balances)} worker balances."
        )
        return len(synced_ticket_ids) + len(synced_movement_ids), True

    except PermissionError:
        logger.warning(f"Cannot rebuild Excel: {excel_path} is locked by another process.")
        return 0, False
    except Exception as e:
        logger.error(f"Error rebuilding Excel: {type(e).__name__}: {e}")
        return 0, False
