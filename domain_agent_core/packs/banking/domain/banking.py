"""Identity verification and human-handoff logic for the banking pack.

Ported from the real ``Livekit Pipeline`` agent's ``src/banking.py``. The
LLM's only job is to *collect* what the caller says and pass it here;
every decision about whether an answer is correct lives in this module
(Python), never in the model. Tool results returned from here are
deliberately minimal - status codes only, never the stored maiden name or
DOB - so no prompt-injection can coax the model into revealing a correct
answer.

Same ``asyncio.Lock``-guarded mutation discipline as the Meridian demo
pack's ``_bank_data.py``, and a pinned ``ZoneInfo`` clock (``BANK_TZ``) for
every timestamp, never the container's naive system clock - the same
discipline ``booking.py``'s and ``prompts.py``'s independently-found
``HOSPITAL_TZ`` bugs established for this codebase.
"""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

BANK_TZ = ZoneInfo("Asia/Karachi")

# Max failed verification attempts allowed per call before we stop
# retrying and auto-queue a human handoff instead (abuse control - don't
# let a caller brute-force different answers).
MAX_CARD_ATTEMPTS = 2


def _now_iso() -> str:
    return datetime.now(BANK_TZ).isoformat()


def normalize(s: str) -> str:
    """Lowercase, trim, and collapse internal whitespace - absorbs STT
    casing/spacing noise without doing any fuzzy/approximate matching on
    the identity value itself."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def parse_date(s: str):
    """Parse a spoken or written date into a ``date`` object, or ``None``
    if unparseable. Callers may say "3rd of July 1998" while the DB stores
    "1998-07-03" - compare dates, never raw strings."""
    s = (s or "").strip()
    if not s:
        return None
    # "3rd" -> "3", "1st" -> "1", etc., so month-name parsing isn't
    # tripped up by ordinals.
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)
    try:
        from dateutil import parser as dtp

        return dtp.parse(cleaned, fuzzy=True).date()
    except Exception:  # noqa: BLE001 - fall through to strict formats below
        pass
    for fmt in (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%B %d %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%d %b %Y",
    ):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _digits_last4(s: str) -> str:
    """Extract the last 4 digits from whatever the caller gave for the
    card ("ends in 4242")."""
    digits = re.sub(r"\D", "", s or "")
    return digits[-4:]


def _register_failure(
    conn, session: dict, customer_id, granular_reason: str, card_last4: str, failed_fields: list[str]
) -> dict:
    """Record a failed verification attempt and either offer a retry or,
    once the per-call limit is hit, auto-queue a human handoff. The
    customer-facing reason is always the generic "verification_failed" -
    we never reveal whether the card even exists (anti-enumeration) or
    which specific field was wrong; that granular detail goes only to the
    audit log."""
    from domain_agent_core.packs.banking.domain import db

    session["failed_card_attempts"] = session.get("failed_card_attempts", 0) + 1
    attempts = session["failed_card_attempts"]
    db.write_audit(
        conn,
        customer_id,
        "verification_failed",
        f"attempt={attempts} reason={granular_reason} card_last4={card_last4} "
        f"failed_fields={','.join(failed_fields) or 'none'}",
    )

    if attempts >= MAX_CARD_ATTEMPTS:
        ticket_id = db.create_handoff_ticket(
            conn,
            customer_id,
            reason="card block: identity verification failed twice",
            created_at=_now_iso(),
        )
        db.write_audit(
            conn, customer_id, "handoff_created", f"ticket={ticket_id} trigger=verification_limit"
        )
        return {"status": "handed_off", "reason": "verification_failed", "ticket_id": ticket_id}

    return {
        "status": "declined",
        "reason": "verification_failed",
        "attempts_remaining": MAX_CARD_ATTEMPTS - attempts,
    }


def verify_and_block_card(conn, session: dict, args: dict) -> dict:
    """Verify the caller against the customer record and block the card
    if they match. ``session`` is the per-call state dict; it carries the
    failed-attempt counter."""
    from domain_agent_core.packs.banking.domain import db

    card_last4 = _digits_last4(args.get("card_last4", ""))
    customer = db.fetch_customer_by_card_last4(conn, card_last4)

    # Unknown card is treated as a failed attempt, and reported
    # identically to a mismatch so a caller can't probe which card
    # numbers exist.
    if not customer:
        return _register_failure(conn, session, None, "not_found", card_last4, ["card_last4"])

    name_ok = normalize(customer["mother_maiden_name"]) == normalize(args.get("mother_maiden_name", ""))
    caller_dob = parse_date(args.get("dob", ""))
    stored_dob = parse_date(customer["dob"])
    dob_ok = caller_dob is not None and caller_dob == stored_dob

    if not (name_ok and dob_ok):
        failed = ([] if name_ok else ["mother_maiden_name"]) + ([] if dob_ok else ["dob"])
        return _register_failure(
            conn, session, customer["customer_id"], "verification_failed", card_last4, failed
        )

    db.set_card_status(conn, customer["customer_id"], "blocked")
    db.write_audit(conn, customer["customer_id"], "card_blocked", f"card_last4={card_last4}")
    session["failed_card_attempts"] = 0  # clean slate after a successful verification
    return {"status": "blocked", "card_last4": card_last4}


def queue_handoff(conn, args: dict) -> dict:
    """Log a callback request for a human representative and return a
    ticket id."""
    from domain_agent_core.packs.banking.domain import db

    reason = (args.get("reason") or "").strip() or "customer requested a human representative"
    customer_id = args.get("customer_id") or None
    ticket_id = db.create_handoff_ticket(conn, customer_id, reason, created_at=_now_iso())
    db.write_audit(conn, customer_id, "handoff_requested", f"ticket={ticket_id}")
    return {"status": "queued", "ticket_id": ticket_id}
