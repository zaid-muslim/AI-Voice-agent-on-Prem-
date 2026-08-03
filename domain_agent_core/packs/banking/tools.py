"""Banking domain tools - the second proof point.

Every tool is backed by real (demo) account/transaction data
(``_bank_data.py``), never something the LLM could assert on its own -
see ``DETERMINISTIC`` below, enforced by ``core/tool_registry.py`` because
this pack's compliance profile (``pci_dss_glba``) requires it.
"""

from __future__ import annotations

import asyncio

from livekit.agents import RunContext, function_tool
from loguru import logger

from domain_agent_core.core.turn_filler import run_with_filler
from domain_agent_core.packs.banking import _bank_data

DEFAULT_FILLERS = {
    "check_balance": "One moment, let me pull up your account.",
    "transfer_funds": "Just a second while I process that transfer.",
    "dispute_transaction": "Let me look into that transaction for you.",
    "report_fraud": "I'm flagging this right away, one moment.",
    "find_branch": "Let me find the nearest branch for you.",
}

# Every banking tool is backed by real account/transaction data, never an
# LLM-asserted fact - required by this pack's pci_dss_glba compliance
# profile (core/tool_registry.check_deterministic_backing()).
DETERMINISTIC = {
    "check_balance",
    "transfer_funds",
    "dispute_transaction",
    "report_fraud",
    "find_branch",
}

_transfer_lock = asyncio.Lock()


@function_tool
async def check_balance(context: RunContext, account_last4: str) -> dict:
    """Check an account's current balance. Never state a balance you were
    not given by this tool's own result.

    Args:
        account_last4: The last 4 digits of the caller's account number -
            never ask for or repeat the full account number aloud.
    """
    agent = context.session.current_agent
    await agent.remember(account_last4=account_last4)

    async def _lookup() -> dict:
        account = _bank_data.find_account(account_last4)
        if account is None:
            return {
                "status": "not_found",
                "message": f"I don't see an account ending in {account_last4}.",
            }
        if account.frozen:
            return {
                "status": "frozen",
                "message": "This account is currently frozen pending a fraud "
                "review. I can't share balance details until that's resolved.",
            }
        return {
            "status": "ok",
            "account_last4": account.last4,
            "balance": account.balance,
            "message": f"The balance on the account ending in {account.last4} "
            f"is ${account.balance:,.2f}.",
        }

    result = await run_with_filler(
        context.session, _lookup(), filler=DEFAULT_FILLERS["check_balance"]
    )
    logger.info(f"tool result: check_balance -> status={result.get('status')!r}")
    await agent.record_tool_call("check_balance", {"account_last4": account_last4}, result)
    return result


@function_tool
async def transfer_funds(
    context: RunContext,
    from_account_last4: str,
    to_account_last4: str,
    amount: float,
    memo: str | None = None,
) -> dict:
    """Transfer funds between two accounts. Confirm the amount and both
    account endings back to the caller before calling this.

    Args:
        from_account_last4: Last 4 digits of the source account.
        to_account_last4: Last 4 digits of the destination account.
        amount: Dollar amount to transfer, confirmed with the caller.
        memo: Optional transfer memo/description.
    """
    agent = context.session.current_agent
    await agent.remember(amount=str(amount))

    async def _transfer() -> dict:
        if amount >= _bank_data.VERIFICATION_THRESHOLD:
            return {
                "status": "requires_verification",
                "message": f"Transfers of ${amount:,.2f} or more need "
                "additional verification before I can complete them. "
                "This deployment doesn't have a live second-factor step "
                "wired in yet - tell the caller a banker will follow up.",
            }
        async with _transfer_lock:
            source = _bank_data.find_account(from_account_last4)
            dest = _bank_data.find_account(to_account_last4)
            if source is None or dest is None:
                missing = from_account_last4 if source is None else to_account_last4
                return {
                    "status": "not_found",
                    "message": f"I don't see an account ending in {missing}.",
                }
            if source.frozen:
                return {
                    "status": "frozen",
                    "message": "The source account is frozen pending a fraud "
                    "review and can't send transfers right now.",
                }
            if source.balance < amount:
                return {
                    "status": "insufficient_funds",
                    "message": f"The account ending in {source.last4} doesn't "
                    f"have enough available balance for a ${amount:,.2f} transfer.",
                }
            source.balance -= amount
            dest.balance += amount
            txn = _bank_data.record_transaction(
                source, -amount, memo or f"Transfer to {dest.last4}"
            )
            _bank_data.record_transaction(dest, amount, memo or f"Transfer from {source.last4}")
            return {
                "status": "completed",
                "transaction_id": txn["transaction_id"],
                "message": f"Done - ${amount:,.2f} transferred from the account "
                f"ending in {source.last4} to the one ending in {dest.last4}. "
                f"Confirmation code {txn['transaction_id']}.",
            }

    result = await run_with_filler(
        context.session, _transfer(), filler=DEFAULT_FILLERS["transfer_funds"]
    )
    logger.info(f"tool result: transfer_funds -> status={result.get('status')!r}")
    await agent.record_tool_call(
        "transfer_funds",
        {
            "from_account_last4": from_account_last4,
            "to_account_last4": to_account_last4,
            "amount": amount,
            "memo": memo,
        },
        result,
    )
    return result


@function_tool
async def dispute_transaction(context: RunContext, transaction_id: str, reason: str) -> dict:
    """Open a dispute on a past transaction, looked up by its transaction id.

    Args:
        transaction_id: The transaction's confirmation code.
        reason: The caller's stated reason for disputing it.
    """
    agent = context.session.current_agent
    await agent.remember(transaction_id=transaction_id)

    async def _dispute() -> dict:
        found = _bank_data.find_transaction(transaction_id)
        if found is None:
            return {
                "status": "not_found",
                "message": f"I don't see a transaction with id {transaction_id}.",
            }
        _account, txn = found
        txn["disputed"] = True
        txn["dispute_reason"] = reason
        return {
            "status": "disputed",
            "transaction_id": transaction_id,
            "message": f"Transaction {transaction_id} has been flagged for "
            "dispute review. A specialist will follow up within 5 business days.",
        }

    result = await run_with_filler(
        context.session, _dispute(), filler=DEFAULT_FILLERS["dispute_transaction"]
    )
    logger.info(f"tool result: dispute_transaction -> status={result.get('status')!r}")
    await agent.record_tool_call(
        "dispute_transaction", {"transaction_id": transaction_id, "reason": reason}, result
    )
    return result


@function_tool
async def report_fraud(context: RunContext, account_last4: str, description: str) -> dict:
    """Report suspected fraud on an account and freeze it immediately.
    Call this the moment a caller describes unauthorized activity - don't
    wait for them to explicitly ask to freeze anything.

    Args:
        account_last4: Last 4 digits of the affected account.
        description: What the caller described.
    """
    agent = context.session.current_agent

    async def _freeze() -> dict:
        account = _bank_data.find_account(account_last4)
        if account is None:
            return {
                "status": "not_found",
                "message": f"I don't see an account ending in {account_last4}.",
            }
        account.frozen = True
        return {
            "status": "frozen",
            "account_last4": account.last4,
            "message": f"The account ending in {account.last4} has been "
            "frozen and flagged for fraud review. A specialist will "
            "contact the account holder shortly.",
        }

    result = await run_with_filler(
        context.session, _freeze(), filler=DEFAULT_FILLERS["report_fraud"]
    )
    logger.info(f"tool result: report_fraud -> status={result.get('status')!r}")
    await agent.record_tool_call(
        "report_fraud", {"account_last4": account_last4, "description": description}, result
    )
    return result


@function_tool
async def find_branch(context: RunContext, query: str) -> dict:
    """Look up branch locations, hours, or general bank policy
    information. Do NOT use this for balance or transaction questions.

    Args:
        query: The caller's question, rephrased as a short search query.
    """
    agent = context.session.current_agent
    rag_engine = agent.assembled.rag_engine
    if rag_engine is None:
        result = {"status": "not_found", "message": "No branch information is configured."}
    else:
        result = await run_with_filler(
            context.session, rag_engine.search(query), filler=DEFAULT_FILLERS["find_branch"]
        )
    await agent.record_tool_call("find_branch", {"query": query}, result)
    return result
