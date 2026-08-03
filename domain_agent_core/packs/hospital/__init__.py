"""Hospital domain pack - Phase 0 regression baseline.

Imports ``app/hospital_core/*`` directly (via the same ``sys.path``-based
shim ``app/compat.py`` uses) so this pack's behavior is bit-for-bit
identical to today's ``RiversideReceptionist`` - the whole point of this
pack is to be the regression safety net proving the new core runtime
didn't change anything before ``banking`` (a genuinely new pack) is built.
"""

from __future__ import annotations
