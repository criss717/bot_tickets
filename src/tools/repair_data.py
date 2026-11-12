"""Maintenance script to repair and sanitize existing database tickets and rebuild Excel."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import load_settings
from db.repository import Repository
from excel.writer import rebuild_excel_from_db
from ocr.extractor import clean_fecha, clean_nif, clean_total

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def repair_database(repo: Repository, excel_path: Path) -> None:
    """Repair dirty historical tickets and rebuild the Excel spreadsheet."""
    logger.info("Starting database repair...")

    with repo._connect() as conn:
        rows = conn.execute("SELECT * FROM tickets ORDER BY id ASC").fetchall()
        logger.info(f"Found {len(rows)} tickets in database.")

        for row in rows:
            tid = row["id"]
            empresa = row["empresa"]
            nif = row["nif"] if "nif" in row.keys() else ""
            desc = row["descripcion"]
            total = row["total"]
            fecha = row["fecha"]

            # Specific known fixes
            if tid == 5:
                # Ticket #5: handwritten Mesón La Casilla 24'00 misread as 24400.0
                total = 24.0
                fecha = "21/05/2024"
                empresa = 'Mesón "La Casilla" S.L.U.'
                logger.info(f"Fixed Ticket #5 -> total=24.00 €, fecha=21/05/2024")
            elif tid == 7 and fecha.upper() in ("DD/MM/YYYY", ""):
                # Ticket #7: copied placeholder from previous prompt
                fecha = "21/05/2024"
                logger.info(f"Fixed Ticket #7 -> fecha=21/05/2024")

            # General sanitization
            new_nif = clean_nif(nif)
            new_total = clean_total(total)
            new_fecha = clean_fecha(fecha)

            conn.execute(
                """UPDATE tickets
                   SET empresa = ?, nif = ?, descripcion = ?, total = ?, fecha = ?, synced_to_excel = 0
                   WHERE id = ?""",
                (empresa, new_nif, desc, new_total, new_fecha, tid),
            )

    logger.info("Database records sanitized.")

    # Rebuild Excel file
    synced, ok = rebuild_excel_from_db(excel_path, repo)
    if ok:
        logger.info(f"✅ Excel rebuild complete: {synced} tickets written to {excel_path}.")
    else:
        logger.warning(f"⚠️ Could not rebuild Excel: file is locked. Run /sincronizar after closing Excel.")


if __name__ == "__main__":
    settings = load_settings()
    repo = Repository(settings.db_path)
    repair_database(repo, settings.excel_path)
