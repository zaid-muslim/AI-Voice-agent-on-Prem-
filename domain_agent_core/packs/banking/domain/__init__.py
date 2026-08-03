"""Pack-local domain layer for the banking pack - the platform's first
genuinely self-contained, pack-local SQLite domain layer (see
``PLAN_real_banking_domain_pack.md``). ``db.py`` owns the schema/connection,
``banking.py`` owns the verification/handoff decisions; both are ported
from the real ``Livekit Pipeline`` agent's ``src/db.py``/``src/banking.py``.
"""

from __future__ import annotations
