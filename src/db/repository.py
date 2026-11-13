"""SQLite repository for users, tickets, and cash movements."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from db.models import CashMovement, MonthlyWorkerSummary, Ticket, User, UserBalance
from ocr.dedup import companies_match, hamming_distance


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER UNIQUE NOT NULL,
    name TEXT NOT NULL,
    role TEXT DEFAULT 'trabajador',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    empresa TEXT NOT NULL,
    nif TEXT DEFAULT '',
    descripcion TEXT NOT NULL,
    total REAL NOT NULL,
    fecha TEXT NOT NULL,
    synced_to_excel INTEGER DEFAULT 0,
    image_sha256 TEXT DEFAULT '',
    image_dhash TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS cash_movements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    created_by_id INTEGER NOT NULL,
    movement_type TEXT NOT NULL,
    amount REAL NOT NULL,
    note TEXT DEFAULT '',
    fecha TEXT NOT NULL,
    synced_to_excel INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (created_by_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_users_telegram_id ON users(telegram_id);
CREATE INDEX IF NOT EXISTS idx_tickets_user_id ON tickets(user_id);
CREATE INDEX IF NOT EXISTS idx_movements_user_id ON cash_movements(user_id);
"""


class Repository:
    """SQLite repository for bot data."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

            # Safe migration for existing tickets table: synced_to_excel & dedup hashes
            cursor = conn.execute("PRAGMA table_info(tickets)")
            existing_ticket_columns = {row["name"] for row in cursor.fetchall()}
            if "synced_to_excel" not in existing_ticket_columns:
                conn.execute("ALTER TABLE tickets ADD COLUMN synced_to_excel INTEGER DEFAULT 0")
            if "image_sha256" not in existing_ticket_columns:
                conn.execute("ALTER TABLE tickets ADD COLUMN image_sha256 TEXT DEFAULT ''")
            if "image_dhash" not in existing_ticket_columns:
                conn.execute("ALTER TABLE tickets ADD COLUMN image_dhash TEXT DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_synced ON tickets(synced_to_excel)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_sha256 ON tickets(image_sha256)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_fecha_total ON tickets(fecha, total)")

            # Safe migration for existing users table: role
            cursor = conn.execute("PRAGMA table_info(users)")
            existing_user_columns = {row["name"] for row in cursor.fetchall()}
            if "role" not in existing_user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'trabajador'")

            # Safe migration for cash_movements indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_movements_synced ON cash_movements(synced_to_excel)")

    @staticmethod
    def _row_to_ticket(row: sqlite3.Row) -> Ticket:
        """Convert a SQLite row from `tickets` into a Ticket dataclass."""
        keys = row.keys()
        return Ticket(
            id=row["id"],
            user_id=row["user_id"],
            image_path=row["image_path"],
            empresa=row["empresa"],
            nif=row["nif"] if "nif" in keys else "",
            descripcion=row["descripcion"],
            total=row["total"],
            fecha=row["fecha"],
            synced_to_excel=bool(row["synced_to_excel"]) if "synced_to_excel" in keys else False,
            image_sha256=row["image_sha256"] if "image_sha256" in keys and row["image_sha256"] else "",
            image_dhash=row["image_dhash"] if "image_dhash" in keys and row["image_dhash"] else "",
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_movement(row: sqlite3.Row) -> CashMovement:
        """Convert a SQLite row from `cash_movements` into a CashMovement dataclass."""
        keys = row.keys()
        return CashMovement(
            id=row["id"],
            user_id=row["user_id"],
            created_by_id=row["created_by_id"],
            movement_type=row["movement_type"],
            amount=row["amount"],
            note=row["note"] if "note" in keys else "",
            fecha=row["fecha"],
            synced_to_excel=bool(row["synced_to_excel"]),
            created_at=row["created_at"],
        )

    # ── Users ─────────────────────────────────────────────────────────────

    def create_user(self, telegram_id: int, name: str, role: str = "trabajador") -> User:
        """Create a new user. Raises if telegram_id already exists."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users (telegram_id, name, role) VALUES (?, ?, ?)",
                (telegram_id, name, role),
            )
            return User(
                id=cursor.lastrowid,
                telegram_id=telegram_id,
                name=name,
                role=role,
            )

    def get_user_by_telegram_id(self, telegram_id: int) -> User | None:
        """Find a user by their Telegram ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE telegram_id = ?",
                (telegram_id,),
            ).fetchone()
            if row is None:
                return None
            return User(
                id=row["id"],
                telegram_id=row["telegram_id"],
                name=row["name"],
                role=row["role"] if "role" in row.keys() and row["role"] else "trabajador",
                created_at=row["created_at"],
            )

    def get_user_by_id(self, user_id: int) -> User | None:
        """Find a user by their internal ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                return None
            return User(
                id=row["id"],
                telegram_id=row["telegram_id"],
                name=row["name"],
                role=row["role"] if "role" in row.keys() and row["role"] else "trabajador",
                created_at=row["created_at"],
            )

    def set_user_role(self, telegram_id_or_id: int, role: str) -> None:
        """Update role for a user by telegram_id or internal ID."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET role = ? WHERE telegram_id = ? OR id = ?",
                (role, telegram_id_or_id, telegram_id_or_id),
            )

    def sync_admin_roles(self, admin_telegram_ids: tuple[int, ...] | set[int]) -> tuple[int, int]:
        """Synchronize DB roles so ADMIN_TELEGRAM_IDS is the single source of truth.

        Promotes real users in admin_telegram_ids to 'admin' and demotes anyone else
        who still had role='admin' in SQLite back to 'trabajador'.
        Returns (promoted_count, demoted_count).
        """
        admin_set = {tid for tid in admin_telegram_ids if tid > 0}
        promoted = 0
        demoted = 0
        with self._connect() as conn:
            rows = conn.execute("SELECT id, telegram_id, role FROM users").fetchall()
            for row in rows:
                tid = row["telegram_id"]
                current_role = row["role"] or "trabajador"
                should_be_admin = tid in admin_set
                if should_be_admin and current_role != "admin":
                    conn.execute("UPDATE users SET role = 'admin' WHERE id = ?", (row["id"],))
                    promoted += 1
                elif not should_be_admin and current_role == "admin":
                    conn.execute("UPDATE users SET role = 'trabajador' WHERE id = ?", (row["id"],))
                    demoted += 1
        return promoted, demoted

    def get_all_users(self) -> list[User]:
        """Get all registered users ordered by ID."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY id ASC").fetchall()
            return [
                User(
                    id=row["id"],
                    telegram_id=row["telegram_id"],
                    name=row["name"],
                    role=row["role"] if "role" in row.keys() and row["role"] else "trabajador",
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def get_admins(self) -> list[User]:
        """Get all users with admin role."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM users WHERE role = 'admin' ORDER BY id ASC").fetchall()
            return [
                User(
                    id=row["id"],
                    telegram_id=row["telegram_id"],
                    name=row["name"],
                    role="admin",
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def create_dummy_worker(self, name: str) -> User:
        """Create a mock/test worker with a negative telegram_id for testing without a second phone."""
        with self._connect() as conn:
            row = conn.execute("SELECT MIN(telegram_id) as min_id FROM users WHERE telegram_id < 0").fetchone()
            min_id = row["min_id"] if row and row["min_id"] is not None else -1000
            next_dummy_id = min_id - 1
            cursor = conn.execute(
                "INSERT INTO users (telegram_id, name, role) VALUES (?, ?, 'trabajador')",
                (next_dummy_id, name),
            )
            return User(
                id=cursor.lastrowid,
                telegram_id=next_dummy_id,
                name=name,
                role="trabajador",
            )

    def update_user_name(self, user_id: int, new_name: str) -> None:
        """Update display name for an existing user."""
        with self._connect() as conn:
            conn.execute("UPDATE users SET name = ? WHERE id = ?", (new_name.strip(), user_id))

    def delete_dummy_worker(self, user_id: int) -> bool:
        """Delete a test worker (telegram_id < 0) if they have zero tickets and zero movements."""
        with self._connect() as conn:
            row = conn.execute("SELECT telegram_id FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None or row["telegram_id"] >= 0:
                return False
            cnt_t = conn.execute("SELECT COUNT(*) as c FROM tickets WHERE user_id = ?", (user_id,)).fetchone()["c"]
            cnt_m = conn.execute("SELECT COUNT(*) as c FROM cash_movements WHERE user_id = ?", (user_id,)).fetchone()["c"]
            if cnt_t > 0 or cnt_m > 0:
                return False
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return True

    # ── Tickets ───────────────────────────────────────────────────────────

    def create_ticket(
        self,
        user_id: int,
        image_path: str,
        empresa: str,
        nif: str,
        descripcion: str,
        total: float,
        fecha: str,
        synced_to_excel: bool = False,
        image_sha256: str = "",
        image_dhash: str = "",
    ) -> Ticket:
        """Create a new ticket record."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO tickets
                   (user_id, image_path, empresa, nif, descripcion, total, fecha, synced_to_excel, image_sha256, image_dhash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, image_path, empresa, nif, descripcion, total, fecha, int(synced_to_excel), image_sha256, image_dhash),
            )
            return Ticket(
                id=cursor.lastrowid,
                user_id=user_id,
                image_path=image_path,
                empresa=empresa,
                nif=nif,
                descripcion=descripcion,
                total=total,
                fecha=fecha,
                synced_to_excel=synced_to_excel,
                image_sha256=image_sha256,
                image_dhash=image_dhash,
            )

    def get_ticket_by_id(self, ticket_id: int) -> Ticket | None:
        """Find a ticket by its internal ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tickets WHERE id = ?",
                (ticket_id,),
            ).fetchone()
            return self._row_to_ticket(row) if row is not None else None

    def get_tickets_by_user(self, user_id: int) -> list[Ticket]:
        """Get all tickets for a specific user."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE user_id = ? ORDER BY id DESC",
                (user_id,),
            ).fetchall()
            return [self._row_to_ticket(row) for row in rows]

    def get_recent_tickets(self, user_id: int | None = None, limit: int = 8) -> list[tuple[Ticket, str]]:
        """Get recent tickets (all users or specific user) ordered newest first."""
        with self._connect() as conn:
            if user_id is not None:
                rows = conn.execute(
                    """SELECT t.*, u.name as user_name FROM tickets t
                       JOIN users u ON t.user_id = u.id
                       WHERE t.user_id = ?
                       ORDER BY t.id DESC LIMIT ?""",
                    (user_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT t.*, u.name as user_name FROM tickets t
                       JOIN users u ON t.user_id = u.id
                       ORDER BY t.id DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [(self._row_to_ticket(row), row["user_name"]) for row in rows]

    def delete_ticket(self, ticket_id: int) -> Ticket | None:
        """Delete a ticket from DB and return the removed Ticket (or None if missing)."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            if row is None:
                return None
            ticket = self._row_to_ticket(row)
            conn.execute("DELETE FROM tickets WHERE id = ?", (ticket_id,))
            return ticket

    def find_duplicate_by_image_hash(
        self,
        sha256: str,
        dhash: str = "",
        max_hamming: int = 4,
        days_window: int = 90,
    ) -> tuple[Ticket, str, str] | None:
        """Check globally if an identical or visually near-identical photo already exists.

        Uses indexed SHA-256 lookup first, then bounds perceptual dHash comparison
        by `created_at` (`days_window`) to keep query latency constant on large DBs.
        """
        with self._connect() as conn:
            if sha256:
                row = conn.execute(
                    """SELECT t.*, u.name as user_name
                       FROM tickets t
                       JOIN users u ON t.user_id = u.id
                       WHERE t.image_sha256 = ?
                       ORDER BY t.id DESC LIMIT 1""",
                    (sha256,),
                ).fetchone()
                if row is not None:
                    return (self._row_to_ticket(row), row["user_name"], "exact")

            if dhash:
                rows = conn.execute(
                    """SELECT t.*, u.name as user_name
                       FROM tickets t
                       JOIN users u ON t.user_id = u.id
                       WHERE t.image_dhash != ''
                         AND t.created_at >= datetime('now', ?)
                       ORDER BY t.id DESC
                       LIMIT 1000""",
                    (f"-{int(days_window)} days",),
                ).fetchall()
                best: tuple[int, sqlite3.Row] | None = None
                for row in rows:
                    dist = hamming_distance(dhash, row["image_dhash"])
                    if dist <= max_hamming and (best is None or dist < best[0]):
                        best = (dist, row)
                if best is not None:
                    return (self._row_to_ticket(best[1]), best[1]["user_name"], "visual")
        return None

    def find_duplicate_by_data(
        self,
        empresa: str,
        total: float,
        fecha: str,
        days_window: int = 90,
    ) -> tuple[Ticket, str] | None:
        """Check globally via composite (fecha, total) index if a matching receipt exists."""
        if total <= 0.0 or not fecha:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT t.*, u.name as user_name
                   FROM tickets t
                   JOIN users u ON t.user_id = u.id
                   WHERE t.fecha = ?
                     AND ROUND(t.total, 2) = ROUND(?, 2)
                     AND t.created_at >= datetime('now', ?)
                   ORDER BY t.id DESC
                   LIMIT 25""",
                (fecha, float(total), f"-{int(days_window)} days"),
            ).fetchall()
            for row in rows:
                if companies_match(row["empresa"], empresa):
                    return (self._row_to_ticket(row), row["user_name"])
        return None

    def count_tickets_by_user(self, user_id: int) -> int:
        """Count tickets for a user."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM tickets WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return row["cnt"]

    def get_unsynced_tickets(self) -> list[tuple[Ticket, str]]:
        """Get all tickets that have not yet been synchronized to Excel.

        Returns:
            List of (Ticket, worker_name) tuples ordered by ticket ID.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT t.*, u.name as user_name
                   FROM tickets t
                   JOIN users u ON t.user_id = u.id
                   WHERE t.synced_to_excel = 0
                   ORDER BY t.id ASC""",
            ).fetchall()
            return [(self._row_to_ticket(row), row["user_name"]) for row in rows]

    def get_all_tickets_with_users(self) -> list[tuple[Ticket, str]]:
        """Get all tickets with user names ordered by ID."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT t.*, u.name as user_name
                   FROM tickets t
                   JOIN users u ON t.user_id = u.id
                   ORDER BY t.id ASC""",
            ).fetchall()
            return [(self._row_to_ticket(row), row["user_name"]) for row in rows]

    def mark_tickets_synced(self, ticket_ids: list[int]) -> None:
        """Mark a list of tickets as synchronized to Excel."""
        if not ticket_ids:
            return
        placeholders = ",".join("?" for _ in ticket_ids)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE tickets SET synced_to_excel = 1 WHERE id IN ({placeholders})",
                ticket_ids,
            )

    def update_ticket(self, ticket_id: int, **fields: Any) -> None:
        """Update arbitrary fields of a ticket."""
        if not fields:
            return
        allowed_columns = {"empresa", "nif", "descripcion", "total", "fecha", "synced_to_excel"}
        set_clauses = []
        values = []
        for col, val in fields.items():
            if col not in allowed_columns:
                raise ValueError(f"Column {col} is not allowed for update")
            set_clauses.append(f"{col} = ?")
            values.append(int(val) if isinstance(val, bool) else val)
        values.append(ticket_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE tickets SET {', '.join(set_clauses)} WHERE id = ?",
                values,
            )

    # ── Cash Movements ────────────────────────────────────────────────────

    def create_cash_movement(
        self,
        user_id: int,
        created_by_id: int,
        movement_type: str,
        amount: float,
        note: str = "",
        fecha: str | None = None,
    ) -> CashMovement:
        """Record a cash movement (ASIGNACION, DESASIGNACION, DEVOLUCION, APORTE_PROPIO)."""
        valid_types = {"ASIGNACION", "DESASIGNACION", "DEVOLUCION", "APORTE_PROPIO"}
        if movement_type not in valid_types:
            raise ValueError(f"Invalid movement_type: {movement_type}. Must be one of {valid_types}")

        date_str = fecha or datetime.now().strftime("%d/%m/%Y")
        clean_amount = round(float(amount), 2)

        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO cash_movements
                   (user_id, created_by_id, movement_type, amount, note, fecha, synced_to_excel)
                   VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (user_id, created_by_id, movement_type, clean_amount, note, date_str),
            )
            return CashMovement(
                id=cursor.lastrowid,
                user_id=user_id,
                created_by_id=created_by_id,
                movement_type=movement_type,
                amount=clean_amount,
                note=note,
                fecha=date_str,
                synced_to_excel=False,
            )

    def get_unsynced_movements(self) -> list[tuple[CashMovement, str, str]]:
        """Get all cash movements not yet synchronized to Excel.

        Returns:
            List of (movement, worker_name, creator_name) tuples ordered by movement ID.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT m.*, u.name as worker_name, c.name as creator_name
                   FROM cash_movements m
                   JOIN users u ON m.user_id = u.id
                   JOIN users c ON m.created_by_id = c.id
                   WHERE m.synced_to_excel = 0
                   ORDER BY m.id ASC""",
            ).fetchall()
            return [(self._row_to_movement(row), row["worker_name"], row["creator_name"]) for row in rows]

    def get_all_movements_with_users(self) -> list[tuple[CashMovement, str, str]]:
        """Get all cash movements with user and creator names."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT m.*, u.name as worker_name, c.name as creator_name
                   FROM cash_movements m
                   JOIN users u ON m.user_id = u.id
                   JOIN users c ON m.created_by_id = c.id
                   ORDER BY m.id ASC""",
            ).fetchall()
            return [(self._row_to_movement(row), row["worker_name"], row["creator_name"]) for row in rows]

    def get_recent_movements(
        self,
        user_id: int | None = None,
        limit: int = 8,
    ) -> list[tuple[CashMovement, str, str]]:
        """Get recent cash movements (optionally filtered by worker) newest first."""
        with self._connect() as conn:
            if user_id is not None:
                rows = conn.execute(
                    """SELECT m.*, u.name as worker_name, c.name as creator_name
                       FROM cash_movements m
                       JOIN users u ON m.user_id = u.id
                       JOIN users c ON m.created_by_id = c.id
                       WHERE m.user_id = ?
                       ORDER BY m.id DESC LIMIT ?""",
                    (user_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT m.*, u.name as worker_name, c.name as creator_name
                       FROM cash_movements m
                       JOIN users u ON m.user_id = u.id
                       JOIN users c ON m.created_by_id = c.id
                       ORDER BY m.id DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [(self._row_to_movement(row), row["worker_name"], row["creator_name"]) for row in rows]

    def get_movement_by_id(self, movement_id: int) -> tuple[CashMovement, str, str] | None:
        """Fetch a single cash movement along with worker and creator names."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT m.*, u.name as worker_name, c.name as creator_name
                   FROM cash_movements m
                   JOIN users u ON m.user_id = u.id
                   JOIN users c ON m.created_by_id = c.id
                   WHERE m.id = ?""",
                (movement_id,),
            ).fetchone()
            if row is None:
                return None
            return (self._row_to_movement(row), row["worker_name"], row["creator_name"])

    def delete_cash_movement(self, movement_id: int) -> CashMovement | None:
        """Delete a cash movement by ID and return it (or None if not found)."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cash_movements WHERE id = ?", (movement_id,)).fetchone()
            if row is None:
                return None
            mov = self._row_to_movement(row)
            conn.execute("DELETE FROM cash_movements WHERE id = ?", (movement_id,))
            return mov

    def mark_movements_synced(self, movement_ids: list[int]) -> None:
        """Mark cash movements as synchronized to Excel."""
        if not movement_ids:
            return
        placeholders = ",".join("?" for _ in movement_ids)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE cash_movements SET synced_to_excel = 1 WHERE id IN ({placeholders})",
                movement_ids,
            )

    # ── Financial Balance & Accounting ────────────────────────────────────

    def get_user_balance(self, user_id: int, year_month: str | None = None) -> UserBalance:
        """Calculate the comprehensive accounting balance for a user with monthly breakdown.

        Option A (Submission Date): Monthly activity is grouped by `strftime('%Y-%m', created_at)`
        in the system, ensuring that current month cash flow reflects tickets and payments
        handled in the ongoing period.

        Accounting Rule:
            Saldo Neto Vivo = Total Asignado - Total Gastado en Tickets Reales - Total Devuelto (NUNCA se reinicia)
            Saldo Mes Anterior = Saldo Neto Vivo acumulado antes del mes seleccionado
        """
        user = self.get_user_by_id(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")

        target_ym = year_month or datetime.now().strftime("%Y-%m")
        try:
            dt_m = datetime.strptime(target_ym, "%Y-%m")
            month_label = dt_m.strftime("%m/%Y")
        except ValueError:
            month_label = target_ym

        with self._connect() as conn:
            # 1. Historical totals (all-time)
            row_asig = conn.execute(
                """SELECT COALESCE(SUM(
                       CASE
                           WHEN movement_type = 'ASIGNACION' THEN amount
                           WHEN movement_type = 'DESASIGNACION' THEN -amount
                           ELSE 0.0
                       END
                   ), 0.0) as s FROM cash_movements WHERE user_id = ?""",
                (user_id,),
            ).fetchone()
            total_assigned = float(row_asig["s"])

            row_spent = conn.execute(
                "SELECT COALESCE(SUM(total), 0.0) as s FROM tickets WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            total_spent = float(row_spent["s"])

            row_ret = conn.execute(
                "SELECT COALESCE(SUM(amount), 0.0) as s FROM cash_movements WHERE user_id = ? AND movement_type = 'DEVOLUCION'",
                (user_id,),
            ).fetchone()
            total_returned = float(row_ret["s"])

            row_pock = conn.execute(
                "SELECT COALESCE(SUM(amount), 0.0) as s FROM cash_movements WHERE user_id = ? AND movement_type = 'APORTE_PROPIO'",
                (user_id,),
            ).fetchone()
            total_own_pocket = float(row_pock["s"])

            # 2. Monthly activity (target_ym)
            row_m_asig = conn.execute(
                """SELECT COALESCE(SUM(
                       CASE
                           WHEN movement_type = 'ASIGNACION' THEN amount
                           WHEN movement_type = 'DESASIGNACION' THEN -amount
                           ELSE 0.0
                       END
                   ), 0.0) as s FROM cash_movements
                   WHERE user_id = ? AND strftime('%Y-%m', created_at) = ?""",
                (user_id, target_ym),
            ).fetchone()
            month_assigned = float(row_m_asig["s"])

            row_m_spent = conn.execute(
                """SELECT COALESCE(SUM(total), 0.0) as s, COUNT(*) as cnt FROM tickets
                   WHERE user_id = ? AND strftime('%Y-%m', created_at) = ?""",
                (user_id, target_ym),
            ).fetchone()
            month_spent = float(row_m_spent["s"])
            month_tickets_count = int(row_m_spent["cnt"])

            row_m_ret = conn.execute(
                """SELECT COALESCE(SUM(amount), 0.0) as s FROM cash_movements
                   WHERE user_id = ? AND movement_type = 'DEVOLUCION' AND strftime('%Y-%m', created_at) = ?""",
                (user_id, target_ym),
            ).fetchone()
            month_returned = float(row_m_ret["s"])

        net_balance = round(total_assigned - total_spent - total_returned, 2)
        prev_assigned = total_assigned - month_assigned
        prev_spent = total_spent - month_spent
        prev_returned = total_returned - month_returned
        previous_balance = round(prev_assigned - prev_spent - prev_returned, 2)

        if abs(net_balance) < 0.01:
            status_label = "🟢 A PAZ Y SALVO (0.00 €)"
        elif net_balance > 0.00:
            status_label = f"🟡 DEBE A LA JEFA ({net_balance:.2f} €)"
        else:
            status_label = f"🔴 LA JEFA LE DEBE ({abs(net_balance):.2f} €)"

        return UserBalance(
            user_id=user.id or user_id,
            name=user.name,
            role=user.role,
            month_label=month_label,
            month_assigned=round(month_assigned, 2),
            month_spent=round(month_spent, 2),
            month_returned=round(month_returned, 2),
            month_tickets_count=month_tickets_count,
            previous_balance=previous_balance,
            total_assigned=round(total_assigned, 2),
            total_spent=round(total_spent, 2),
            total_returned=round(total_returned, 2),
            total_own_pocket=round(total_own_pocket, 2),
            net_balance=net_balance,
            status_label=status_label,
        )

    def get_all_balances(self, year_month: str | None = None) -> list[UserBalance]:
        """Calculate balances for all registered users."""
        users = self.get_all_users()
        return [self.get_user_balance(u.id, year_month=year_month) for u in users if u.id is not None]

    def get_monthly_summaries(self) -> list[MonthlyWorkerSummary]:
        """Get historical monthly financial aggregations for each worker."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT strftime('%Y-%m', created_at) as ym, user_id
                   FROM (
                       SELECT created_at, user_id FROM tickets
                       UNION ALL
                       SELECT created_at, user_id FROM cash_movements
                   )
                   WHERE ym IS NOT NULL
                   ORDER BY ym DESC, user_id ASC"""
            ).fetchall()

            summaries: list[MonthlyWorkerSummary] = []
            for r in rows:
                ym = r["ym"]
                uid = r["user_id"]
                user = self.get_user_by_id(uid)
                if user is None:
                    continue

                r_asig = conn.execute(
                    """SELECT COALESCE(SUM(
                           CASE
                               WHEN movement_type = 'ASIGNACION' THEN amount
                               WHEN movement_type = 'DESASIGNACION' THEN -amount
                               ELSE 0.0
                           END
                       ), 0.0) as s FROM cash_movements
                       WHERE user_id = ? AND strftime('%Y-%m', created_at) = ?""",
                    (uid, ym),
                ).fetchone()
                asig = float(r_asig["s"])

                r_spent = conn.execute(
                    """SELECT COALESCE(SUM(total), 0.0) as s, COUNT(*) as cnt FROM tickets
                       WHERE user_id = ? AND strftime('%Y-%m', created_at) = ?""",
                    (uid, ym),
                ).fetchone()
                spent = float(r_spent["s"])
                cnt = int(r_spent["cnt"])

                r_ret = conn.execute(
                    """SELECT COALESCE(SUM(amount), 0.0) as s FROM cash_movements
                       WHERE user_id = ? AND movement_type = 'DEVOLUCION' AND strftime('%Y-%m', created_at) = ?""",
                    (uid, ym),
                ).fetchone()
                ret = float(r_ret["s"])

                net_month = round(asig - spent - ret, 2)
                summaries.append(
                    MonthlyWorkerSummary(
                        year_month=ym,
                        user_id=uid,
                        user_name=user.name,
                        role=user.role,
                        tickets_count=cnt,
                        total_assigned=round(asig, 2),
                        total_spent=round(spent, 2),
                        total_returned=round(ret, 2),
                        net_month=net_month,
                    )
                )
            return summaries
