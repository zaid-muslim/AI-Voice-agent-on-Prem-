"""Phase 2 tests: compliance profiles are enforced concretely, not just
described in a prompt.

Run:
    venv/bin/python -m pytest domain_agent_core/tests/test_compliance.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from domain_agent_core.core import audit_log
from domain_agent_core.core.compliance_profiles import get_compliance_policy
from domain_agent_core.core.domain_loader import AgentAssembler


def test_pci_dss_glba_never_writes_a_raw_account_number(tmp_path) -> None:
    policy = get_compliance_policy("pci_dss_glba")
    audit_log.record(
        tmp_path,
        "banking",
        policy,
        event_type="tool_call",
        payload={"tool": "check_balance", "account_number": "4111111111111111"},
    )
    files = audit_log.list_audit_files(tmp_path, "banking")
    assert len(files) == 1
    line = files[0].read_text().strip()
    entry = json.loads(line)
    assert "4111111111111111" not in line
    assert entry["payload"]["account_number"].startswith("<redacted:sha256:")


def test_audit_log_never_rotates_away_old_files(tmp_path) -> None:
    """Deliberately diverges from app/latency_log.py's size-triggered
    rotation - a regulatory audit trail must never silently lose data."""
    policy = get_compliance_policy("hipaa")
    old_file = tmp_path / "hospital_audit_2020-01.jsonl"
    old_file.write_text('{"old": "entry"}\n')

    for _ in range(50):
        audit_log.record(
            tmp_path, "hospital", policy, event_type="tool_call", payload={"n": "x" * 500}
        )

    assert old_file.exists()
    assert old_file.read_text() == '{"old": "entry"}\n'
    files = audit_log.list_audit_files(tmp_path, "hospital")
    assert old_file in files


def test_none_profile_still_writes_standard_level_logs(tmp_path) -> None:
    """"none" profile's audit_log_level is "standard", which still
    writes (unredacted) - only a hypothetical "off" level would skip
    entirely."""
    policy = get_compliance_policy("none")
    audit_log.record(tmp_path, "demo", policy, event_type="tool_call", payload={"x": 1})
    files = audit_log.list_audit_files(tmp_path, "demo")
    assert len(files) == 1


def test_missing_deterministic_set_fails_at_load_time(tmp_path) -> None:
    """A pack whose compliance profile requires deterministic tool
    backing, but whose tools module has no DETERMINISTIC set, must fail
    AgentAssembler.assemble() loudly - a deploy-time error, not a runtime
    surprise."""
    packs_dir = tmp_path / "packs"
    pack_dir = packs_dir / "broken"
    pack_dir.mkdir(parents=True)
    (pack_dir / "persona.jinja2").write_text("hi")
    (pack_dir / "manifest.yaml").write_text(
        "domain_id: broken\n"
        "agent_name: broken-agent\n"
        "timezone: UTC\n"
        "persona: {template_path: persona.jinja2, greeting_template: hi, variables: {}}\n"
        "engines: {llm: {served_model_name: x, source: x}, "
        "stt: {engine: whisper_shared, model: x}, tts: {engine: qwen, model: x}}\n"
        "tools: {module: os.path, allow_list: [join]}\n"
        "safety: {module: os.path}\n"
        "compliance: {profile: hipaa}\n"  # requires deterministic backing; os.path has no DETERMINISTIC
    )

    assembler = AgentAssembler(packs_dir=packs_dir)
    with pytest.raises(ValueError, match="DETERMINISTIC"):
        assembler.assemble("broken")
