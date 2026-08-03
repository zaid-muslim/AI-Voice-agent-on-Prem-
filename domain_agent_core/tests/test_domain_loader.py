"""Phase 0 regression tests for ``AgentAssembler`` + the hospital pack.

Run:
    venv/bin/python -m pytest domain_agent_core/tests/test_domain_loader.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_APP_DIR = _REPO_ROOT / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from domain_agent_core.core import prompt_builder
from domain_agent_core.core.domain_loader import AgentAssembler


def test_hospital_pack_assembles_cleanly() -> None:
    assembled = AgentAssembler().assemble("hospital")
    assert assembled.pack.domain_id == "hospital"
    assert assembled.pack.agent_name == "hospital-receptionist"
    assert assembled.compliance.name == "hipaa"
    assert {t.info.name for t in assembled.tool_callables} == {
        "check_availability",
        "book_appointment",
        "cancel_appointment",
        "update_appointment",
        "search_hospital_info",
    }


def test_hospital_prompt_is_string_identical_to_original() -> None:
    """The literal regression contract: Phase 0's Jinja2 persona template
    must be a byte-for-byte port of app/prompts.py's f-string. Compared
    with compliance_clause="" - the compliance-clause injection is a
    deliberate NEW Phase 2 addition layered on top of this unchanged core
    content, not part of what this test guards."""
    import prompts as original_prompts  # app/prompts.py

    assembled = AgentAssembler().assemble("hospital")
    rendered = prompt_builder.build_system_prompt(
        assembled.persona_template_path,
        assembled.pack.persona.variables,
        assembled.pack.timezone,
        compliance_clause="",
    )
    original = original_prompts.build_system_prompt()
    assert rendered.strip() == original.strip()


def test_hospital_greeting_matches_original() -> None:
    import prompts as original_prompts

    assembled = AgentAssembler().assemble("hospital")
    rendered = prompt_builder.build_greeting_instructions(
        assembled.pack.persona.greeting_template, assembled.pack.persona.variables
    )
    assert rendered.strip() == original_prompts.GREETING_INSTRUCTIONS.strip()


def test_agent_name_collision_is_rejected(tmp_path) -> None:
    """A second pack declaring the same agent_name as an existing one
    must fail loudly at load time, not silently let LiveKit's dispatch
    become ambiguous."""
    packs_dir = tmp_path / "packs"
    manifest_body = (
        "domain_id: {domain_id}\n"
        "agent_name: shared-name\n"
        "timezone: UTC\n"
        "persona: {{template_path: persona.jinja2, greeting_template: hi, variables: {{}}}}\n"
        "engines: {{llm: {{served_model_name: x, source: x}}, "
        "stt: {{engine: whisper_shared, model: x}}, tts: {{engine: qwen, model: x}}}}\n"
        "tools: {{module: os.path, allow_list: [join]}}\n"
        "safety: {{module: os.path}}\n"
        "compliance: {{profile: none}}\n"
    )
    for domain_id in ("pack_a", "pack_b"):
        pack_dir = packs_dir / domain_id
        pack_dir.mkdir(parents=True)
        (pack_dir / "persona.jinja2").write_text("hi")
        (pack_dir / "manifest.yaml").write_text(manifest_body.format(domain_id=domain_id))

    from domain_agent_core.core.domain_loader import AgentAssembler as _Assembler

    assembler = _Assembler(packs_dir=packs_dir)
    try:
        assembler.assemble("pack_a")
        raised = False
    except ValueError as exc:
        raised = True
        assert "shared-name" in str(exc)
        assert "pack_b" in str(exc)
    assert raised, "expected a collision on agent_name to raise ValueError"
