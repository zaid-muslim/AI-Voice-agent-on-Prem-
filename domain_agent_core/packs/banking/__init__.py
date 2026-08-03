"""Banking domain pack - the real HBL telephone-banking agent, ported
from the sibling ``Livekit Pipeline`` repo onto ``domain_agent_core``'s
generic runtime (see ``PLAN_real_banking_domain_pack.md`` at the repo
root). Card-block identity verification and human handoff are backed by
a pack-local SQLite domain layer (``domain/``); business/product Q&A is
grounded in this pack's own KB via ``core/rag_engine.py``.
"""

from __future__ import annotations
