#!/usr/bin/env python3
"""SQLite storage for the banking agent: customer records (for card-block identity
verification), an audit log of every verification attempt, and a queue of human-handoff
callback tickets.

All functions take an explicit connection so the same code backs the live server, the
seed script, and the tests. The server opens one shared connection (see server.py);
sqlite3 with WAL is fine for the single-event-loop access pattern here.
"""
import os
import sqlite3
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "bank.db")

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


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    """Open (creating the parent dir if needed) and return a connection with rows as dicts.
    check_same_thread=False lets the async server share one connection across executor threads."""
    parent = os.path.dirname(path)
    if parent:   # skip for ":memory:" and bare filenames
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def fetch_customer_by_card_last4(conn: sqlite3.Connection, card_last4: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM customers WHERE card_last4 = ? LIMIT 1", (card_last4,)
    ).fetchone()
    return dict(row) if row else None


def set_card_status(conn: sqlite3.Connection, customer_id: str, status: str) -> None:
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
    conn.execute(
        "INSERT INTO audit_log (ts, customer_id, action, detail) VALUES (?, ?, ?, ?)",
        (datetime.now().astimezone().isoformat(), customer_id, action, detail),
    )
    conn.commit()
