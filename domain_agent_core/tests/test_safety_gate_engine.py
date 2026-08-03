"""Phase 0 regression test: the hospital pack's safety policy must be
15/15 identical pass/fail to app/hospital_core/safety.py's own self-test.

Run:
    venv/bin/python -m pytest domain_agent_core/tests/test_safety_gate_engine.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from domain_agent_core.core.safety_gate import run_self_test
from domain_agent_core.packs.hospital.safety_policy import POLICY, TEST_CASES


def test_hospital_safety_policy_self_test_passes() -> None:
    assert run_self_test(POLICY, TEST_CASES, verbose=False) is True


def test_hospital_safety_policy_matches_original_rule_count() -> None:
    sys.path.insert(0, str(_REPO_ROOT / "app" / "hospital_core"))
    import safety as original_safety

    assert len(POLICY.rules) == len(original_safety.RULES)
    assert set(POLICY.escalation_messages) == set(original_safety.ESCALATION_MESSAGE)
