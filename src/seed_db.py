#!/usr/bin/env python3
"""Create the SQLite schema and seed it with fake customers for testing card-block
verification. Safe to re-run: it upserts the sample customers and resets their card
status to active, so you get a clean slate each time.

Run: python3 src/seed_db.py
"""
import db

# Fake customers. dob is stored ISO (YYYY-MM-DD); the verification code parses whatever
# the caller says (e.g. "3rd of July 1998") into a date before comparing, so the stored
# format doesn't have to match how a caller phrases it.
SAMPLE_CUSTOMERS = [
    # customer_id, card_last4, mother_maiden_name, dob, cnic_last4
    ("C001", "4242", "Bibi",     "1990-05-14", "1234"),
    ("C002", "1881", "Sultana",  "1985-11-02", "5678"),
    ("C003", "0007", "Khatoon",  "1998-07-03", "9012"),
]


def seed(conn) -> None:
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


if __name__ == "__main__":
    conn = db.connect()
    seed(conn)
    count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    print(f"Seeded {len(SAMPLE_CUSTOMERS)} customers into {db.DB_PATH} ({count} total).")
    for row in conn.execute(
        "SELECT customer_id, card_last4, mother_maiden_name, dob, card_status FROM customers"
    ):
        print(f"  {row['customer_id']}: card ...{row['card_last4']}, "
              f"maiden={row['mother_maiden_name']}, dob={row['dob']}, {row['card_status']}")
    conn.close()
