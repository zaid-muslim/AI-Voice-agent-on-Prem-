"""Seed data for the banking pack's SQLite domain layer, ported from the
real ``Livekit Pipeline`` agent's ``src/seed_db.py`` - the same four demo
customers (C001-C004), so a fresh checkout has working test accounts
immediately for the card-block verification flow.
"""

from __future__ import annotations

# customer_id, card_last4, mother_maiden_name, dob (ISO), cnic_last4
SAMPLE_CUSTOMERS: list[tuple[str, str, str, str, str]] = [
    ("C001", "4242", "Bibi", "1990-05-14", "1234"),
    ("C002", "1881", "Sultana", "1985-11-02", "5678"),
    ("C003", "0007", "Khatoon", "1998-07-03", "9012"),
    ("C004", "3480", "Amelia", "1992-03-21", "3456"),
]


def seed(conn) -> None:
    """Create the schema (if needed) and upsert every sample customer,
    resetting each to an active card status - safe to call repeatedly."""
    from domain_agent_core.packs.banking.domain import db

    db.init_schema(conn)
    for customer_id, card_last4, maiden, dob, cnic in SAMPLE_CUSTOMERS:
        conn.execute(
            """
            INSERT INTO customers
                (customer_id, card_last4, mother_maiden_name, dob, cnic_last4, card_status)
            VALUES (?, ?, ?, ?, ?, 'active')
            ON CONFLICT(customer_id) DO UPDATE SET
                card_last4         = excluded.card_last4,
                mother_maiden_name = excluded.mother_maiden_name,
                dob                = excluded.dob,
                cnic_last4         = excluded.cnic_last4,
                card_status        = 'active'
            """,
            (customer_id, card_last4, maiden, dob, cnic),
        )
    conn.commit()


def seed_if_empty(conn) -> None:
    """Seed the sample customers only if the ``customers`` table is
    currently empty - so a deployment that has already blocked/modified a
    real seeded row on a live call doesn't get silently reset every
    process restart."""
    from domain_agent_core.packs.banking.domain import db

    db.init_schema(conn)
    count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    if count == 0:
        seed(conn)
