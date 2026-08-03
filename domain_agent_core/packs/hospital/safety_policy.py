"""Hospital safety policy - Phase 0 regression baseline.

Imports ``app/hospital_core/safety.py``'s ``RULES``/``ESCALATION_MESSAGE``
directly (same import shim as ``tools.py``) so this pack's safety-gate
behavior is bit-for-bit identical to today's pipeline - the actual
medical/self-harm regex rules are not re-authored here, only wrapped into
the generic ``core.safety_gate.SafetyPolicy`` shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[3] / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

_CORE_DIR = _APP_DIR / "hospital_core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

import safety as _hospital_safety

from domain_agent_core.core.safety_gate import EmergencyRule, SafetyPolicy

POLICY = SafetyPolicy(
    rules=tuple(
        EmergencyRule(category=r.category, kind=r.kind, patterns=tuple(r.patterns))
        for r in _hospital_safety.RULES
    ),
    escalation_messages=dict(_hospital_safety.ESCALATION_MESSAGE),
)

# Deliberate negatives kept alongside the positives - a naive matcher
# would get these wrong (see hospital_core/safety.py:214-221's identical
# discipline, ported verbatim).
TEST_CASES: list[tuple[str, bool]] = [
    ("I'm having really bad chest pain right now", True),
    ("I think I'm having a heart attack", True),
    ("I can't breathe, please help", True),
    ("my face is drooping and my speech is slurred", True),
    ("this is the worst headache of my life", True),
    ("there's severe bleeding and it won't stop", True),
    ("I took too many pills an hour ago", True),
    ("I want to kill myself", True),
    ("I don't want to be alive anymore", True),
    ("I'm killing time waiting for my appointment", False),
    ("I have a small headache today", False),
    ("can you help me book an appointment with a cardiologist", False),
    ("my chest has been a bit sore after the gym", False),
    ("I'd like to check my medication refill status", False),
    ("what are your visiting hours", False),
]


if __name__ == "__main__":
    from domain_agent_core.core.safety_gate import run_self_test

    run_self_test(POLICY, TEST_CASES)
