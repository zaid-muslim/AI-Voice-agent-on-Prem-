#!/usr/bin/env python3
"""Dump the banking database so you can eyeball it: customers, the verification audit log, and
queued human-handoff tickets. Read-only.

Run: python3 src/show_db.py            (all tables)
     python3 src/show_db.py audit_log  (one table)
"""
import sys

import db

TABLES = ["customers", "audit_log", "handoff_tickets"]


def dump(conn, table: str) -> None:
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    print(f"\n=== {table} ({len(rows)} rows) ===")
    if not rows:
        print("  (empty)")
        return
    cols = rows[0].keys()
    print("  " + " | ".join(cols))
    print("  " + "-" * 60)
    for r in rows:
        print("  " + " | ".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    conn = db.connect()
    wanted = sys.argv[1:] or TABLES
    print(f"Database: {db.DB_PATH}")
    for t in wanted:
        dump(conn, t)
    conn.close()
