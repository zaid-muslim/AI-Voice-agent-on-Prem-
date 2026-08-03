"""Phase 1 tests: the banking pack is the platform's core-thesis proof
point - a genuinely different vertical assembling cleanly on the exact
same core/ runtime, with its own tools/safety/compliance, zero shared
code with the hospital pack beyond core/.

Run:
    venv/bin/python -m pytest domain_agent_core/tests/test_banking_pack.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from domain_agent_core.core.domain_loader import AgentAssembler
from domain_agent_core.core.safety_gate import run_self_test
from domain_agent_core.packs.banking.safety_policy import POLICY, TEST_CASES


def test_banking_pack_assembles_cleanly() -> None:
    assembled = AgentAssembler().assemble("banking")
    assert assembled.pack.domain_id == "banking"
    assert assembled.pack.agent_name == "banking-agent"
    assert assembled.compliance.name == "pci_dss_glba"
    assert {t.info.name for t in assembled.tool_callables} == {
        "check_balance",
        "transfer_funds",
        "dispute_transaction",
        "report_fraud",
        "find_branch",
    }
    # pci_dss_glba's duress category must be wired to silent escalation.
    assert assembled.safety_policy.silent_kinds == {"duress"}
    assert assembled.rag_engine is not None


def test_banking_safety_policy_self_test_passes() -> None:
    assert run_self_test(POLICY, TEST_CASES, verbose=False) is True


def test_banking_and_hospital_agent_names_do_not_collide() -> None:
    assembler = AgentAssembler()
    hospital = assembler.assemble("hospital")
    banking = assembler.assemble("banking")
    assert hospital.pack.agent_name != banking.pack.agent_name


def test_missing_tool_in_allow_list_fails_loudly(tmp_path) -> None:
    """tool_registry.load_tools() must raise, listing every problem, if
    an allow-listed name doesn't exist or isn't a decorated tool."""
    from domain_agent_core.core import tool_registry

    with pytest.raises(ValueError, match="does not exist"):
        tool_registry.load_tools(
            "domain_agent_core.packs.banking.tools",
            ("check_balance", "this_tool_does_not_exist"),
        )


def test_check_balance_is_deterministic_not_llm_asserted() -> None:
    """check_balance must return the tool's own literal data, never
    something a caller/LLM could inject."""
    from domain_agent_core.packs.banking import _bank_data

    account = _bank_data.find_account("4421")
    assert account is not None
    assert account.balance == 2450.32


def test_transfer_funds_requires_verification_above_threshold() -> None:
    from domain_agent_core.packs.banking import _bank_data

    assert _bank_data.VERIFICATION_THRESHOLD == 1000.0
