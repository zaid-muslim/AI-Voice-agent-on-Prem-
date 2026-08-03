"""Tests for the banking pack - the real HBL telephone-banking agent,
ported from the sibling ``Livekit Pipeline`` repo onto this platform's
``core/`` runtime (see ``PLAN_real_banking_domain_pack.md``).

Covers both layers: the pack assembling cleanly on the generic runtime
(tools/safety/compliance/RAG), and the pack-local SQLite domain layer's
card-block verification and human-handoff decisions - ported test-for-test
from that repo's own ``tests/test_banking.py``, since the underlying
verification logic in ``domain/banking.py`` is an as-is port.

Run:
    venv/bin/python -m pytest domain_agent_core/tests/test_banking_pack.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from domain_agent_core.core.domain_loader import AgentAssembler
from domain_agent_core.core.safety_gate import run_self_test
from domain_agent_core.packs.banking.domain import banking, db
from domain_agent_core.packs.banking.safety_policy import POLICY, TEST_CASES


# ── Pack assembly (Phase 1 proof point) ──────────────────────────────────────

def test_banking_pack_assembles_cleanly() -> None:
    assembled = AgentAssembler().assemble("banking")
    assert assembled.pack.domain_id == "banking"
    assert assembled.pack.agent_name == "banking-agent"
    assert assembled.compliance.name == "pci_dss_glba"
    assert {t.info.name for t in assembled.tool_callables} == {
        "block_card",
        "request_human_handoff",
        "search_business_docs",
    }
    # pci_dss_glba's duress category must be wired to silent escalation.
    assert assembled.safety_policy.silent_kinds == {"duress"}
    assert assembled.rag_engine is not None


def test_banking_safety_policy_self_test_passes() -> None:
    assert run_self_test(POLICY, TEST_CASES, verbose=False) is True


def test_banking_and_hospital_agent_names_do_not_collide() -> None:
    assembler = AgentAssembler()
    hospital = assembler.assemble("hospital")
    banking_domain = assembler.assemble("banking")
    assert hospital.pack.agent_name != banking_domain.pack.agent_name


def test_missing_tool_in_allow_list_fails_loudly(tmp_path) -> None:
    """tool_registry.load_tools() must raise, listing every problem, if
    an allow-listed name doesn't exist or isn't a decorated tool."""
    from domain_agent_core.core import tool_registry

    with pytest.raises(ValueError, match="does not exist"):
        tool_registry.load_tools(
            "domain_agent_core.packs.banking.tools",
            ("block_card", "this_tool_does_not_exist"),
        )


def test_rag_grounded_lookup_returns_real_kb_content() -> None:
    """A real product/policy question must be answered from the ported
    KB, not an invented answer."""
    assembled = AgentAssembler().assemble("banking")
    result = asyncio.run(assembled.rag_engine.search("branch hours"))
    assert result["status"] == "ok"
    assert "Monday" in result["answer"]


# ── Domain layer: card-block verification (ported from Livekit Pipeline's
#    tests/test_banking.py, same fixtures/assertions, real schema) ──────────

@pytest.fixture
def conn():
    """Fresh in-memory DB seeded with one known customer per test -
    isolated from the pack's own persistent data/bank.db."""
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


def test_valid_block_correct_answers(conn, session) -> None:
    out = banking.verify_and_block_card(
        conn, session, {"card_last4": "4242", "mother_maiden_name": "Bibi", "dob": "1990-05-14"}
    )
    assert out["status"] == "blocked"
    assert out["card_last4"] == "4242"
    assert card_status(conn, "C001") == "blocked"
    assert session["failed_card_attempts"] == 0


def test_block_tolerates_stt_noise(conn, session) -> None:
    """Casing/spacing on the name and a spoken date phrasing must still
    verify."""
    out = banking.verify_and_block_card(
        conn,
        session,
        {"card_last4": "card ending 4242", "mother_maiden_name": "  bibi ", "dob": "14th of May 1990"},
    )
    assert out["status"] == "blocked"


def test_wrong_maiden_name_declined_with_one_retry(conn, session) -> None:
    out = banking.verify_and_block_card(
        conn, session, {"card_last4": "4242", "mother_maiden_name": "Sultana", "dob": "1990-05-14"}
    )
    assert out["status"] == "declined"
    assert out["reason"] == "verification_failed"
    assert out["attempts_remaining"] == 1
    assert card_status(conn, "C001") == "active"  # not blocked


def test_two_failures_auto_handoff_no_block(conn, session) -> None:
    args_bad = {"card_last4": "4242", "mother_maiden_name": "Wrong", "dob": "1990-05-14"}
    first = banking.verify_and_block_card(conn, session, args_bad)
    assert first["status"] == "declined"
    second = banking.verify_and_block_card(conn, session, args_bad)
    assert second["status"] == "handed_off"
    assert second["reason"] == "verification_failed"
    assert "ticket_id" in second
    assert ticket_count(conn) == 1
    assert card_status(conn, "C001") == "active"  # still never blocked


def test_unknown_card_reported_like_a_mismatch(conn, session) -> None:
    """An unknown card must look identical to a wrong answer (no card
    enumeration)."""
    out = banking.verify_and_block_card(
        conn, session, {"card_last4": "9999", "mother_maiden_name": "Bibi", "dob": "1990-05-14"}
    )
    assert out["status"] == "declined"
    assert out["reason"] == "verification_failed"  # not "not_found"


def test_tool_result_never_leaks_secrets(conn, session) -> None:
    """The dict handed back to the model must not contain the stored
    maiden name or DOB."""
    blob = str(
        banking.verify_and_block_card(
            conn, session, {"card_last4": "4242", "mother_maiden_name": "Nope", "dob": "2000-01-01"}
        )
    )
    assert "Bibi" not in blob
    assert "1990" not in blob


# ── Domain layer: human handoff ───────────────────────────────────────────────

def test_direct_handoff_queues_ticket(conn) -> None:
    out = banking.queue_handoff(conn, {"reason": "wants to talk to a person"})
    assert out["status"] == "queued"
    assert isinstance(out["ticket_id"], int)
    assert ticket_count(conn) == 1


def test_handoff_defaults_reason_when_missing(conn) -> None:
    out = banking.queue_handoff(conn, {})
    assert out["status"] == "queued"
    row = conn.execute(
        "SELECT reason FROM handoff_tickets WHERE id = ?", (out["ticket_id"],)
    ).fetchone()
    assert row["reason"]  # non-empty fallback reason stored


# ── Helpers ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "a,b",
    [
        ("1990-05-14", "14th of May 1990"),
        ("1998-07-03", "3rd of July 1998"),
        ("1985-11-02", "November 2 1985"),
    ],
)
def test_parse_date_equates_spoken_and_iso(a, b) -> None:
    assert banking.parse_date(a) == banking.parse_date(b)


def test_parse_date_rejects_garbage() -> None:
    assert banking.parse_date("not a date") is None
    assert banking.parse_date("") is None
