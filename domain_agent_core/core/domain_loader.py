"""``AgentAssembler``: reads a ``manifest.yaml``, validates it, and builds
an ``AssembledDomain`` - the fully-resolved, ready-to-run bundle a
``BaseDomainAgent`` and ``worker.py`` need. Nothing in this repo currently
loads a behavior-defining config from YAML (``system_config.py``'s JSON is
model-selection only) - this is genuinely new code, not a port.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from domain_agent_core.core import engine_config, kb_store, tool_registry
from domain_agent_core.core.compliance_profiles import (
    CompliancePolicy,
    get_compliance_policy,
)
from domain_agent_core.core.domain_pack import (
    DomainPack,
    EngineSelection,
    FillerConfig,
    KbConfig,
    PersonaConfig,
    RecapConfig,
    SafetyConfig,
    ToolsConfig,
)
from domain_agent_core.core.rag_engine import RagEngine
from domain_agent_core.core.safety_gate import SafetyPolicy

_PACKS_DIR = Path(__file__).resolve().parents[1] / "packs"


@dataclass
class AssembledDomain:
    """Everything a running agent needs, resolved from one manifest.

    Attributes:
        pack: The validated ``DomainPack``.
        tool_callables: Free-function ``@function_tool`` callables,
            resolved via ``tool_registry.load_tools()``, ready to pass as
            ``Agent(tools=...)``.
        safety_policy: The domain's ``SafetyPolicy`` (rules + messages),
            with ``silent_kinds`` populated from the resolved
            ``CompliancePolicy``.
        compliance: The resolved ``CompliancePolicy``.
        rag_engine: A ``RagEngine`` over this domain's KB, or ``None`` if
            ``DomainPack.kb.directory`` is ``None``.
        engines: Current llm/stt/tts engine-config dict (from
            ``engine_config.get_engine_config()``, seeded from the
            manifest's ``engines:`` block).
        persona_template_path: Absolute path to this pack's persona
            template file.
        audit_dir: Absolute path to this pack's audit-log directory.
    """

    pack: DomainPack
    tool_callables: list[Callable]
    safety_policy: SafetyPolicy
    compliance: CompliancePolicy
    rag_engine: RagEngine | None
    engines: dict[str, dict[str, Any]]
    persona_template_path: Path
    audit_dir: Path


def _parse_engine_selection(raw: dict[str, Any]) -> EngineSelection:
    engine = raw.get("engine") or raw.get("served_model_name") or ""
    model = raw.get("model") or raw.get("source") or ""
    return EngineSelection(engine=engine, model=model, display_name=raw.get("display_name", ""))


def _parse_manifest(raw: dict[str, Any]) -> DomainPack:
    persona_raw = raw.get("persona", {})
    tools_raw = raw.get("tools", {})
    safety_raw = raw.get("safety", {})
    filler_raw = raw.get("filler", {})
    recap_raw = raw.get("recap", {})
    kb_raw = raw.get("kb", {})
    compliance_raw = raw.get("compliance", {})
    engines_raw = raw.get("engines", {})

    return DomainPack(
        domain_id=raw.get("domain_id", ""),
        display_name=raw.get("display_name", ""),
        agent_name=raw.get("agent_name", ""),
        timezone=raw.get("timezone", ""),
        persona=PersonaConfig(
            template_path=persona_raw.get("template_path", ""),
            greeting_template=persona_raw.get("greeting_template", ""),
            variables=persona_raw.get("variables", {}) or {},
        ),
        engines={
            slot: _parse_engine_selection(engines_raw.get(slot, {}))
            for slot in ("llm", "stt", "tts")
        },
        tools=ToolsConfig(
            module=tools_raw.get("module", ""),
            allow_list=tuple(tools_raw.get("allow_list", []) or []),
        ),
        safety=SafetyConfig(module=safety_raw.get("module", "")),
        filler=FillerConfig(
            phrases=tuple(filler_raw.get("phrases", []) or []),
            per_tool=filler_raw.get("per_tool", {}) or {},
        ),
        recap=RecapConfig(
            labels=tuple(tuple(pair) for pair in recap_raw.get("labels", []) or [])
        ),
        kb=KbConfig(directory=kb_raw.get("directory")),
        compliance_profile=compliance_raw.get("profile", "none"),
        strip_thinking_channel=bool(raw.get("strip_thinking_channel", True)),
    )


class AgentAssembler:
    """Loads domain packs from ``packs/<domain_id>/manifest.yaml``."""

    def __init__(self, packs_dir: Path = _PACKS_DIR) -> None:
        """Initialize the assembler.

        Args:
            packs_dir: Root directory containing one subdirectory per
                domain pack. Defaults to ``domain_agent_core/packs/``.
        """
        self._packs_dir = packs_dir

    def _all_agent_names(self) -> dict[str, str]:
        """Scan every manifest under ``packs_dir`` for its ``agent_name``,
        to detect cross-pack collisions.

        Returns:
            ``{agent_name: domain_id}`` for every manifest found.
        """
        names: dict[str, str] = {}
        if not self._packs_dir.is_dir():
            return names
        for manifest_path in sorted(self._packs_dir.glob("*/manifest.yaml")):
            with open(manifest_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            agent_name = raw.get("agent_name", "")
            domain_id = raw.get("domain_id", manifest_path.parent.name)
            if agent_name:
                names[agent_name] = domain_id
        return names

    def assemble(self, domain_id: str) -> AssembledDomain:
        """Load, validate, and fully resolve one domain pack.

        Args:
            domain_id: The pack's directory name under ``packs_dir``
                (e.g. "hospital", "banking").

        Returns:
            A ready-to-run ``AssembledDomain``.

        Raises:
            FileNotFoundError: If no ``manifest.yaml`` exists for
                ``domain_id``.
            ValueError: If the manifest fails ``DomainPack.validate()``,
                declares an ``agent_name`` already used by another pack,
                references an unregistered compliance profile, or (for a
                compliance profile requiring it) is missing deterministic-
                tool-backing declarations. Every problem is collected and
                raised together, not just the first.
        """
        pack_dir = self._packs_dir / domain_id
        manifest_path = pack_dir / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest.yaml found for domain {domain_id!r} at {pack_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        pack = _parse_manifest(raw)

        problems = pack.validate()

        other_names = {
            name: owner
            for name, owner in self._all_agent_names().items()
            if owner != domain_id
        }
        if pack.agent_name in other_names:
            problems.append(
                f"agent_name {pack.agent_name!r} is already used by domain "
                f"{other_names[pack.agent_name]!r} - agent_name must be "
                "globally unique across every pack"
            )

        compliance: CompliancePolicy | None = None
        try:
            compliance = get_compliance_policy(pack.compliance_profile)
        except KeyError as exc:
            problems.append(str(exc))

        if compliance is not None and compliance.require_deterministic_tool_backing:
            problems.extend(
                tool_registry.check_deterministic_backing(
                    pack.tools.module, pack.tools.allow_list
                )
            )

        if problems:
            raise ValueError(
                f"Invalid domain pack {domain_id!r}:\n  - " + "\n  - ".join(problems)
            )

        tool_callables = tool_registry.load_tools(pack.tools.module, pack.tools.allow_list)

        safety_module = importlib.import_module(pack.safety.module)
        base_policy: SafetyPolicy = safety_module.POLICY
        safety_policy = SafetyPolicy(
            rules=base_policy.rules,
            escalation_messages=base_policy.escalation_messages,
            silent_kinds=compliance.silent_escalation_categories,
        )

        kb_directory = pack_dir / pack.kb.directory if pack.kb.directory else None
        rag_engine: RagEngine | None = None
        if kb_directory is not None:
            kb_path = kb_store.kb_file_path(kb_directory, pack.domain_id)
            entries = kb_store.load_kb(kb_path, default_entries=[])
            rag_engine = RagEngine(entries)

        defaults = {
            slot: {"engine": sel.engine, "model": sel.model, "display_name": sel.display_name}
            for slot, sel in pack.engines.items()
        }
        # engine_config expects llm's keys named served_model_name/source;
        # the manifest's engines.llm block already uses those names via
        # _parse_engine_selection's engine/model aliasing above.
        engine_cfg_path = engine_config.config_file_path(pack_dir)
        engines = engine_config.get_engine_config(engine_cfg_path, defaults)

        return AssembledDomain(
            pack=pack,
            tool_callables=tool_callables,
            safety_policy=safety_policy,
            compliance=compliance,
            rag_engine=rag_engine,
            engines=engines,
            persona_template_path=pack_dir / pack.persona.template_path,
            audit_dir=pack_dir / "audit_logs",
        )
