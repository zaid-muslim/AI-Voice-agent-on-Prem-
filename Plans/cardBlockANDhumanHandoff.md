# Banking Voice Agent — Next Features Implementation Plan

**Branch:** `banking-agent`
**Current state:** Conversational + informational only (no tool calls, no DB writes)
**This plan covers:** (1) Card/debit blocking with identity verification, (2) Human handoff without a telephony layer

---

## Feature 1: Card/Debit Blocking

### 1.1 Goal
Customer says something like "block my card, I lost it" → agent asks verification questions one at a time → answers are checked against the customer DB → card is blocked if valid, request is declined if not.

### 1.2 Data requirements (new/confirm existing)
Need a customer record accessible to the server process with at minimum:
- `customer_id`
- `card_last4` (or full card ref, masked in logs)
- `mother_maiden_name`
- `dob`
- `cnic_last4` (optional third factor)
- `card_status` (active / blocked)

If this table doesn't exist yet, that's a prerequisite — Fable 5 should either build a small SQLite table + seed script, or wire up to whatever DB the POS/banking backend already uses. **Confirm which before starting.**

### 1.3 New tool definition (added to the existing Ollama tool-calling setup)
```python
{
  "type": "function",
  "function": {
    "name": "block_card",
    "description": "Block a customer's card after identity verification. Only call this once all required verification fields have been collected from the customer in conversation.",
    "parameters": {
      "type": "object",
      "properties": {
        "card_last4": {"type": "string"},
        "mother_maiden_name": {"type": "string"},
        "dob": {"type": "string"}
      },
      "required": ["card_last4", "mother_maiden_name", "dob"]
    }
  }
}
```

### 1.4 System prompt addition
Add an explicit instruction block, something like:

> When a customer requests to block a lost/stolen card, you must first collect the following verification details, one question at a time, in natural conversation:
> 1. Last 4 digits of the card
> 2. Mother's maiden name (on file)
> 3. Date of birth
>
> Do not skip questions. Do not confirm or hint whether an answer is correct. Once all three are collected, call `block_card` with exactly what the customer said — do not judge validity yourself, the system will verify it.

**Important:** the LLM's job is only to *collect* answers, never to *judge* them. All matching logic lives in Python.

### 1.5 Verification handler (Python, not LLM)
```python
async def handle_block_card(args, db):
    customer = await db.fetch_customer_by_card_last4(args["card_last4"])
    if not customer:
        return {"status": "declined", "reason": "not_found"}

    name_ok = normalize(customer["mother_maiden_name"]) == normalize(args["mother_maiden_name"])
    dob_ok = parse_date(customer["dob"]) == parse_date(args["dob"])

    if not (name_ok and dob_ok):
        return {"status": "declined", "reason": "verification_failed"}

    await db.set_card_status(customer["customer_id"], "blocked")
    await audit_log.write(customer["customer_id"], "card_blocked", redact(args))
    return {"status": "blocked", "card_last4": args["card_last4"]}
```

`normalize()` = lowercase + strip whitespace, for the name (STT casing/spacing noise).
`parse_date()` = parse both into a date object before comparing (STT may say "3rd of July 1998" vs DB storing "1998-07-03" — don't string-compare raw text).

**Do not fuzzy-match the name itself.** Formatting noise is fine to normalize; approximate/fuzzy matching on the actual identity fields defeats the point of verification.

### 1.6 Security / abuse controls
- **Attempt limit**: max 2 failed verification attempts per call → after that, auto-trigger human handoff (see Feature 2) instead of letting the customer keep retrying different answers.
- **Never expose DB values to the LLM.** The tool result passed back into the model should only ever be `{"status": "blocked"}` or `{"status": "declined", "reason": "..."}` — never the actual stored maiden name/DOB, or a prompt-injection style request could get it to reveal correct answers.
- **Audit log every attempt**, success or failure, with timestamp, customer_id (if found), and which fields failed — needed for fraud review later.
- **Mask card numbers in logs** (`card_last4` only, never full PAN even if somehow captured).

### 1.7 Conversation flow (state machine, roughly)
```
IDLE
  → customer requests card block → COLLECTING (ask card_last4)
COLLECTING
  → ask mother_maiden_name
  → ask dob
  → all collected → call block_card tool
VERIFY_RESULT
  → blocked → confirm to customer, end flow
  → declined (attempt 1) → tell customer info didn't match, retry once
  → declined (attempt 2) → hand off to human (Feature 2)
```
This can live as explicit state in the per-call asyncio task (matches how you're already tracking turn state / barge-in), or just be handled implicitly by the LLM's tool-calling loop with a retry counter passed in context. Fable 5 should pick whichever fits the existing turn-loop code better — flag this as a design decision to make while reading `server.py`.

---

## Feature 2: Human Handoff (no telephony layer yet)

### 2.1 Problem
Currently, if the customer asks for a human/representative, the agent recites the bank's contact number — which is the number this very voice agent answers on. Infinite loop.

### 2.2 Fix, given no telephony/PBX integration yet
Since there's no real transfer mechanism to hand off to, the fix is: **stop pretending to transfer, and instead log a callback request.**

New tool:
```python
{
  "type": "function",
  "function": {
    "name": "request_human_handoff",
    "description": "Request a callback from a human representative. Use when the customer explicitly asks for a human, or after repeated failed card verification.",
    "parameters": {
      "type": "object",
      "properties": {
        "reason": {"type": "string"},
        "customer_id": {"type": "string"}
      },
      "required": ["reason"]
    }
  }
}
```

Handler writes to a simple queue (new SQLite table or even a JSON/CSV file for now, given this is a prototype):
```python
async def handle_handoff(args, db):
    ticket_id = await db.create_handoff_ticket(
        customer_id=args.get("customer_id"),
        reason=args["reason"],
        created_at=now(),
        status="pending"
    )
    return {"status": "queued", "ticket_id": ticket_id}
```

### 2.3 System prompt addition
> You do not have the ability to transfer calls to a human agent right now, and you must never give out the bank's phone number as a way to reach a representative — that number connects back to you. If a customer asks for a human, or if identity verification fails twice, call `request_human_handoff` and tell the customer a representative will contact them back within [X business hours/timeframe] — do not give any phone number.

Fill in `[X]` with whatever turnaround is realistic — this is a business/ops decision, not a technical one, so confirm with whoever owns the actual callback process before hardcoding a promise.

### 2.4 Trigger conditions
- Explicit request ("talk to a person", "representative", "human agent")
- Card verification failing twice (see 1.6)
- Possibly: any request outside the agent's known scope (future-proofing, not needed for v1)

### 2.5 Open item — future telephony
Once a real telephony/PBX layer exists, `request_human_handoff` becomes a real SIP transfer instead of a queued ticket — same tool interface, different backend implementation. Worth designing the tool's interface now so it doesn't need to change later, just the handler body.

---

## Shared implementation notes for Fable 5

- Both new tools slot into the same Ollama `/api/chat` tool-calling mechanism already used for `web_search` in `server.py` — same message loop, just new function names in the tools list and a dispatch branch for each.
- Tool handlers should be `async` and awaited inline in the existing per-turn asyncio task, same pattern as everything else in the turn loop (so barge-in/cancellation still works mid-verification).
- All DB/logging is currently undefined — first task should be confirming or creating the schema (customer table + handoff ticket table) before wiring the tools to it.
- Test cases to write before calling this done:
  - Valid card block (correct answers) → blocked
  - Invalid card block (wrong maiden name) → declined, one retry offered
  - Two failed attempts → auto handoff, no phone number given
  - Direct "let me talk to a human" → immediate handoff, no phone number given
  - Barge-in during verification question → doesn't corrupt collected state
