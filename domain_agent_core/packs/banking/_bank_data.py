"""In-memory demo account/transaction store for the banking pack.

Mirrors ``app/hospital_core/booking.py``'s core data-layer disciplines,
scaled down to an in-memory dict since this pack's job is to prove the
platform's architecture on a second, genuinely different vertical, not to
ship a production core-banking system:

- ``asyncio.Lock``-guarded check-and-set for every mutation, same
  atomicity discipline as ``booking.py``'s SQLite ``UNIQUE`` constraint
  (a race can't produce two conflicting transfers).
- A pinned ``ZoneInfo`` clock (``BANK_TZ``), never the container's naive
  system clock, for every "when did this happen" timestamp - the same
  discipline ``booking.py``'s and ``prompts.py``'s independently-found
  ``HOSPITAL_TZ`` bugs established for this codebase.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

BANK_TZ = ZoneInfo("America/New_York")

# Amounts at or above this require a second-factor verification step
# before completing - the seam for a future OTP integration (see
# transfer_funds()'s "requires_verification" branch in tools.py).
VERIFICATION_THRESHOLD = 1000.0


@dataclass
class Account:
    """One demo bank account.

    Attributes:
        last4: The account's last 4 digits (used as its lookup key -
            never the full account number, per the pci_dss_glba
            compliance profile).
        owner_name: The account holder's name.
        balance: Current balance in dollars.
        frozen: Whether ``report_fraud`` has frozen this account.
        transactions: Recent transaction records.
    """

    last4: str
    owner_name: str
    balance: float
    frozen: bool = False
    transactions: list[dict] = field(default_factory=list)


_ACCOUNTS: dict[str, Account] = {
    "4421": Account(last4="4421", owner_name="Alex Rivera", balance=2450.32),
    "7789": Account(last4="7789", owner_name="Jordan Lee", balance=612.10),
    "1190": Account(last4="1190", owner_name="Sam Okafor", balance=15320.75),
}


def _now() -> datetime:
    return datetime.now(BANK_TZ)


def find_account(last4: str) -> Account | None:
    """Look up a demo account by its last 4 digits."""
    return _ACCOUNTS.get(last4.strip())


def find_transaction(transaction_id: str) -> tuple[Account, dict] | None:
    """Find a transaction and its owning account by transaction id."""
    for account in _ACCOUNTS.values():
        for txn in account.transactions:
            if txn["transaction_id"] == transaction_id:
                return account, txn
    return None


def record_transaction(account: Account, amount: float, description: str) -> dict:
    """Append a new transaction to an account, timestamped in ``BANK_TZ``."""
    txn = {
        "transaction_id": uuid.uuid4().hex[:10],
        "amount": amount,
        "description": description,
        "timestamp": _now().isoformat(),
    }
    account.transactions.append(txn)
    return txn
