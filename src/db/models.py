"""Database models for users, tickets, and cash movements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """A Telegram user registered in the bot."""

    telegram_id: int
    name: str
    role: str = "trabajador"  # 'admin' or 'trabajador'
    created_at: datetime | None = None
    id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Ticket:
    """A processed ticket/receipt."""

    user_id: int
    image_path: str
    empresa: str
    nif: str
    descripcion: str
    total: float
    fecha: str
    synced_to_excel: bool = False
    image_sha256: str = ""
    image_dhash: str = ""
    created_at: datetime | None = None
    id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class CashMovement:
    """A cash movement (assignment, unassignment, return, or out-of-pocket cash)."""

    user_id: int
    created_by_id: int
    movement_type: str  # 'ASIGNACION', 'DESASIGNACION', 'DEVOLUCION', 'APORTE_PROPIO'
    amount: float
    note: str = ""
    fecha: str = ""
    synced_to_excel: bool = False
    created_at: datetime | None = None
    id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()
        if not self.fecha:
            self.fecha = self.created_at.strftime("%d/%m/%Y")


@dataclass
class UserBalance:
    """Summary of financial balance for a worker with monthly breakdown."""

    user_id: int
    name: str
    role: str
    month_label: str  # e.g. "11/2026"
    month_assigned: float
    month_spent: float
    month_returned: float
    month_tickets_count: int
    previous_balance: float  # Saldo arrastrado de meses previos
    total_assigned: float   # Acumulado histórico total
    total_spent: float      # Acumulado histórico total
    total_returned: float   # Acumulado histórico total
    total_own_pocket: float # Total fondos propios declarados
    net_balance: float      # Saldo neto vivo (NUNCA se reinicia)
    status_label: str       # 🟢 A PAZ Y SALVO, 🟡 DEBE A LA JEFA, 🔴 LA JEFA LE DEBE


@dataclass
class MonthlyWorkerSummary:
    """Monthly financial aggregation per worker for Excel historical close."""

    year_month: str  # "YYYY-MM"
    user_id: int
    user_name: str
    role: str
    tickets_count: int
    total_assigned: float
    total_spent: float
    total_returned: float
    net_month: float
