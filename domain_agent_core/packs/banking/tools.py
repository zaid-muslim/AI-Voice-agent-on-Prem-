"""Banking domain tools - the real HBL telephone-banking agent, ported
from ``Livekit Pipeline``'s ``src/banking.py``/``src/rag.py`` onto this
platform's tool contract.

Every tool is backed by real (pack-local SQLite or KB-grounded) data,
never something the LLM could assert on its own - see ``DETERMINISTIC``
below, enforced by ``core/tool_registry.py`` because this pack's
compliance profile (``pci_dss_glba``) requires it.

This agent's scope is deliberately narrow, same as the real system: it
verifies a caller's identity and blocks a lost/stolen card, queues a
human callback, and answers business/product questions from the bank's
own documents. It does NOT check balances or move money - there is no
``check_balance``/``transfer_funds`` tool here, unlike the synthetic
Meridian pack this one replaces, because the real agent never had that
capability either.
"""

from __future__ import annotations

import asyncio

from livekit.agents import RunContext, function_tool
from loguru import logger

from domain_agent_core.core.turn_filler import run_with_filler
from domain_agent_core.packs.banking.domain import banking as domain_banking
from domain_agent_core.packs.banking.domain import db as domain_db
from domain_agent_core.packs.banking.domain import seed as domain_seed

DEFAULT_FILLERS = {
    "block_card": "One moment, let me verify a few details first.",
    "request_human_handoff": "One moment, I'm connecting you with a representative.",
    "search_business_docs": "Let me check that for you.",
}

# Every banking tool is backed by real pack-local data (SQLite identity
# records, or the domain's own KB via RagEngine), never an LLM-asserted
# fact - required by this pack's pci_dss_glba compliance profile
# (core/tool_registry.check_deterministic_backing()).
DETERMINISTIC = {
    "block_card",
    "request_human_handoff",
    "search_business_docs",
}

# One shared connection for the pack's lifetime, same "single shared
# connection, WAL mode" pattern the real system's db.py documents - the
# lock below guards the read-then-write verification sequence, not
# individual statements (sqlite3 already serializes those).
_conn = domain_db.connect()
domain_seed.seed_if_empty(_conn)
_verify_lock = asyncio.Lock()


def _get_verification_session(agent) -> dict:
    """Per-call failed-attempt counter, scoped to this call's agent
    instance (a fresh ``BaseDomainAgent`` is built per call - see
    ``worker.py``'s ``entrypoint()`` - so an instance attribute is
    call-scoped for free, no separate session store needed)."""
    if not hasattr(agent, "_banking_verification_session"):
        agent._banking_verification_session = {}
    return agent._banking_verification_session


@function_tool
async def block_card(
    context: RunContext, card_last4: str, mother_maiden_name: str, dob: str
) -> dict:
    """Verify the caller's identity and block their card if it matches.
    Ask for the card's last 4 digits, their mother's maiden name, and
    their date of birth, in that order, before calling this. Never state
    that a card was blocked or that identity was verified unless this
    tool's own result says so.

    Args:
        card_last4: The last 4 digits of the caller's card - never ask
            for or repeat the full card number aloud.
        mother_maiden_name: The caller's stated mother's maiden name.
        dob: The caller's stated date of birth, however they said it
            (e.g. "3rd of July 1998") - this tool parses it itself.
    """
    agent = context.session.current_agent
    await agent.remember(card_last4=card_last4)
    session = _get_verification_session(agent)

    async def _verify() -> dict:
        async with _verify_lock:
            return domain_banking.verify_and_block_card(
                _conn,
                session,
                {
                    "card_last4": card_last4,
                    "mother_maiden_name": mother_maiden_name,
                    "dob": dob,
                },
            )

    result = await run_with_filler(
        context.session, _verify(), filler=DEFAULT_FILLERS["block_card"]
    )
    logger.info(f"tool result: block_card -> status={result.get('status')!r}")
    await agent.record_tool_call(
        "block_card",
        {"card_last4": card_last4, "mother_maiden_name": "[redacted]", "dob": "[redacted]"},
        result,
    )
    return result


@function_tool
async def request_human_handoff(context: RunContext, reason: str) -> dict:
    """Queue a callback from a human representative. Call this whenever
    the caller explicitly asks for a human, or whenever a request is
    outside what you can help with over the phone.

    Args:
        reason: A short description of why the caller needs a human
            representative.
    """
    agent = context.session.current_agent

    async def _queue() -> dict:
        return domain_banking.queue_handoff(_conn, {"reason": reason})

    result = await run_with_filler(
        context.session, _queue(), filler=DEFAULT_FILLERS["request_human_handoff"]
    )
    logger.info(f"tool result: request_human_handoff -> status={result.get('status')!r}")
    await agent.record_tool_call("request_human_handoff", {"reason": reason}, result)
    return result


@function_tool
async def search_business_docs(context: RunContext, query: str) -> dict:
    """Look up bank products, fees, branch hours, or general policy
    information. Do NOT use this for identity verification or card
    blocking. If you genuinely don't know something about the bank, use
    this tool before telling the caller you don't have the information.

    Args:
        query: The caller's question, rephrased as a short search query.
    """
    agent = context.session.current_agent
    rag_engine = agent.assembled.rag_engine
    if rag_engine is None:
        result = {"status": "not_found", "message": "No business information is configured."}
    else:
        result = await run_with_filler(
            context.session, rag_engine.search(query), filler=DEFAULT_FILLERS["search_business_docs"]
        )
    await agent.record_tool_call("search_business_docs", {"query": query}, result)
    return result
