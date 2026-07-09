"""Tests for the card-block verification and human-handoff logic (src/banking.py).

These cover the deterministic Python that the plan insists must own every verification
decision — no LLM involved. Run: pytest tests/ -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import banking  # noqa: E402
import db        # noqa: E402


@pytest.fixture
def conn():
    """Fresh in-memory DB seeded with one known customer per test."""
    c = db.connect(":memory:")
    db.init_schema(c)
    c.execute(
        "INSERT INTO customers (customer_id, card_last4, mother_maiden_name, dob, cnic_last4, card_status)"
        " VALUES (?, ?, ?, ?, ?, 'active')",
        ("C001", "4242", "Bibi", "1990-05-14", "1234"),
    )
    c.commit()
    yield c
    c.close()


@pytest.fixture
def session():
    return {"failed_card_attempts": 0}


def card_status(conn, customer_id):
    return conn.execute(
        "SELECT card_status FROM customers WHERE customer_id = ?", (customer_id,)
    ).fetchone()["card_status"]


def ticket_count(conn):
    return conn.execute("SELECT COUNT(*) FROM handoff_tickets").fetchone()[0]


# ── Feature 1: card blocking ────────────────────────────────────────────────────

def test_valid_block_correct_answers(conn, session):
    out = banking.verify_and_block_card(conn, session, {
        "card_last4": "4242", "mother_maiden_name": "Bibi", "dob": "1990-05-14",
    })
    assert out["status"] == "blocked"
    assert out["card_last4"] == "4242"
    assert card_status(conn, "C001") == "blocked"
    assert session["failed_card_attempts"] == 0


def test_block_tolerates_stt_noise(conn, session):
    """Casing/spacing on the name and a spoken date phrasing must still verify."""
    out = banking.verify_and_block_card(conn, session, {
        "card_last4": "card ending 4242", "mother_maiden_name": "  bibi ", "dob": "14th of May 1990",
    })
    assert out["status"] == "blocked"


def test_wrong_maiden_name_declined_with_one_retry(conn, session):
    out = banking.verify_and_block_card(conn, session, {
        "card_last4": "4242", "mother_maiden_name": "Sultana", "dob": "1990-05-14",
    })
    assert out["status"] == "declined"
    assert out["reason"] == "verification_failed"
    assert out["attempts_remaining"] == 1
    assert card_status(conn, "C001") == "active"   # not blocked


def test_two_failures_auto_handoff_no_block(conn, session):
    args_bad = {"card_last4": "4242", "mother_maiden_name": "Wrong", "dob": "1990-05-14"}
    first = banking.verify_and_block_card(conn, session, args_bad)
    assert first["status"] == "declined"
    second = banking.verify_and_block_card(conn, session, args_bad)
    assert second["status"] == "handed_off"
    assert second["reason"] == "verification_failed"
    assert "ticket_id" in second
    assert ticket_count(conn) == 1
    assert card_status(conn, "C001") == "active"   # still never blocked


def test_unknown_card_reported_like_a_mismatch(conn, session):
    """An unknown card must look identical to a wrong answer (no card enumeration)."""
    out = banking.verify_and_block_card(conn, session, {
        "card_last4": "9999", "mother_maiden_name": "Bibi", "dob": "1990-05-14",
    })
    assert out["status"] == "declined"
    assert out["reason"] == "verification_failed"   # not "not_found"


def test_tool_result_never_leaks_secrets(conn, session):
    """The dict handed back to the model must not contain the stored maiden name or DOB."""
    blob = str(banking.verify_and_block_card(conn, session, {
        "card_last4": "4242", "mother_maiden_name": "Nope", "dob": "2000-01-01",
    }))
    assert "Bibi" not in blob
    assert "1990" not in blob


# ── Feature 2: human handoff ─────────────────────────────────────────────────────

def test_direct_handoff_queues_ticket(conn):
    out = banking.queue_handoff(conn, {"reason": "wants to talk to a person"})
    assert out["status"] == "queued"
    assert isinstance(out["ticket_id"], int)
    assert ticket_count(conn) == 1


def test_handoff_defaults_reason_when_missing(conn):
    out = banking.queue_handoff(conn, {})
    assert out["status"] == "queued"
    row = conn.execute("SELECT reason FROM handoff_tickets WHERE id = ?", (out["ticket_id"],)).fetchone()
    assert row["reason"]   # non-empty fallback reason stored


# ── Helpers ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("1990-05-14", "14th of May 1990"),
    ("1998-07-03", "3rd of July 1998"),
    ("1985-11-02", "November 2 1985"),
])
def test_parse_date_equates_spoken_and_iso(a, b):
    assert banking.parse_date(a) == banking.parse_date(b)


def test_parse_date_rejects_garbage():
    assert banking.parse_date("not a date") is None
    assert banking.parse_date("") is None
