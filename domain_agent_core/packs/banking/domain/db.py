"""SQLite storage for the banking pack: customer records (for card-block
identity verification), an audit log of every verification attempt, and a
queue of human-handoff callback tickets.

Ported from the real ``Livekit Pipeline`` agent's ``src/db.py`` almost
as-is (same ``customers``/``audit_log``/``handoff_tickets`` schema),
pointed at a pack-local file instead of that repo's own ``data/bank.db``.
The real schema's ``rag_chunks`` table is dropped - this pack's RAG goes
through ``core/kb_store.py`` + ``core/rag_engine.py`` (a tool call, per
this port's architecture decision) rather than the real system's
always-on fastembed/SQLite index, so there is no chunk table to own here.

All functions take an explicit connection so the same code backs the
worker, the seed step, and the tests.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "bank.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id        TEXT PRIMARY KEY,
    card_last4         TEXT NOT NULL,
    mother_maiden_name TEXT NOT NULL,
    dob                TEXT NOT NULL,          -- ISO YYYY-MM-DD
    cnic_last4         TEXT,
    card_status        TEXT NOT NULL DEFAULT 'active'   -- active | blocked
);
CREATE INDEX IF NOT EXISTS idx_customers_card_last4 ON customers(card_last4);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    customer_id TEXT,                          -- NULL when no matching customer was found
    action      TEXT NOT NULL,                 -- e.g. card_blocked, verification_failed
    detail      TEXT                           -- free-text, must never contain the stored secrets
);

CREATE TABLE IF NOT EXISTS handoff_tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    reason      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
);
"""


def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open (creating the parent dir if needed) and return a connection
    with rows as dicts. ``check_same_thread=False`` lets the async worker
    share one connection across executor threads.

    Args:
        path: Database file path, or ``":memory:"`` for tests.

    Returns:
        An open ``sqlite3.Connection`` with ``row_factory`` set.
    """
    path = str(path)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create every table in ``SCHEMA`` if it doesn't already exist."""
    conn.executescript(SCHEMA)
    conn.commit()


def fetch_customer_by_card_last4(conn: sqlite3.Connection, card_last4: str) -> dict | None:
    """Look up a customer by their card's last 4 digits."""
    row = conn.execute(
        "SELECT * FROM customers WHERE card_last4 = ? LIMIT 1", (card_last4,)
    ).fetchone()
    return dict(row) if row else None


def set_card_status(conn: sqlite3.Connection, customer_id: str, status: str) -> None:
    """Update a customer's card status (``"active"`` or ``"blocked"``)."""
    conn.execute(
        "UPDATE customers SET card_status = ? WHERE customer_id = ?", (status, customer_id)
    )
    conn.commit()


def create_handoff_ticket(
    conn: sqlite3.Connection,
    customer_id: str | None,
    reason: str,
    created_at: str | None = None,
    status: str = "pending",
) -> int:
    """Insert a new human-handoff ticket and return its id."""
    created_at = created_at or datetime.now().astimezone().isoformat()
    cur = conn.execute(
        "INSERT INTO handoff_tickets (customer_id, reason, created_at, status) VALUES (?, ?, ?, ?)",
        (customer_id, reason, created_at, status),
    )
    conn.commit()
    return cur.lastrowid


def write_audit(
    conn: sqlite3.Connection, customer_id: str | None, action: str, detail: str = ""
) -> None:
    """Append one audit-log row, timestamped at write time."""
    conn.execute(
        "INSERT INTO audit_log (ts, customer_id, action, detail) VALUES (?, ?, ?, ?)",
        (datetime.now().astimezone().isoformat(), customer_id, action, detail),
    )
    conn.commit()
