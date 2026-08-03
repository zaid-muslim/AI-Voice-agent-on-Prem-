"""Typed schema for a domain-pack manifest (``packs/<domain>/manifest.yaml``).

A ``DomainPack`` is the validated, in-memory form of one domain's manifest
file - persona template location/variables, engine selection, tool
allow-list, safety-policy module, filler phrases, recap field labels, KB
directory, and compliance profile. ``domain_loader.AgentAssembler`` reads
the YAML, builds one of these, and validates it before anything else in
the pipeline touches it.

Uses stdlib ``dataclasses`` + a hand-written ``validate() -> list[str]``,
matching the "collect every problem into one list, then raise once"
convention already established in ``app/system_config.py``'s
``save_config()`` - this repo has no pydantic usage anywhere, and a small,
rarely-changed manifest schema doesn't justify introducing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EngineSelection:
    """One engine's configuration - reuses ``system_config.py``'s exact
    ``{engine/served_model_name, model/source, display_name}`` shape.

    Attributes:
        engine: Which plugin/backend to construct (e.g. "whisper_shared",
            "qwen"). For the LLM slot this is instead the served model
            name, kept under the same field for schema uniformity.
        model: The specific model/checkpoint/voice for ``engine``. For the
            LLM slot this is the model source path (local dir or HF repo).
        display_name: Human-readable label, shown in any future admin UI.
    """

    engine: str
    model: str
    display_name: str = ""


@dataclass(frozen=True)
class PersonaConfig:
    """Persona/system-prompt configuration.

    Attributes:
        template_path: Path to a Jinja2 template file, relative to the
            pack's own directory (e.g. "persona.jinja2").
        greeting_template: Short instruction string used as the opening
            turn's ``generate_reply(instructions=...)`` argument - not a
            full template, just a Jinja2-renderable string.
        variables: Values injected into both templates (e.g.
            ``hospital_name``, ``emergency_number``).
    """

    template_path: str
    greeting_template: str
    variables: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolsConfig:
    """Which tools this domain exposes to the LLM.

    Attributes:
        module: Dotted import path of the pack's tools module (e.g.
            ``"domain_agent_core.packs.banking.tools"``).
        allow_list: Names that must resolve to ``@function_tool``-decorated
            callables in ``module`` - enforced by ``tool_registry.py``.
    """

    module: str
    allow_list: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SafetyConfig:
    """Which safety policy this domain uses.

    Attributes:
        module: Dotted import path of the pack's safety-policy module.
            That module must expose a module-level ``POLICY:
            core.safety_gate.SafetyPolicy`` instance.
    """

    module: str


@dataclass(frozen=True)
class FillerConfig:
    """Turn-filler phrase sets for this domain.

    Attributes:
        phrases: Round-robin generic phrases spoken if the agent hasn't
            started a real reply within the turn-filler delay.
        per_tool: Tool-name -> phrase spoken if that SPECIFIC tool call
            runs slower than its own (much shorter) threshold.
    """

    phrases: tuple[str, ...] = field(default_factory=tuple)
    per_tool: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RecapConfig:
    """Which caller-stated facts survive chat-context truncation.

    Attributes:
        labels: Ordered ``(field_key, human_label)`` pairs, e.g.
            ``("patient_name", "patient name")``.
    """

    labels: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class KbConfig:
    """Knowledge-base location for this domain's RAG engine.

    Attributes:
        directory: Path (relative to the pack's own directory) to a
            directory containing a ``<domain_id>_kb.json`` file, or
            ``None`` if this domain has no RAG needs (or, for Phase 0's
            hospital pack, reuses ``app/hospital_core/hospital_kb.py``
            directly instead of a pack-local KB).
    """

    directory: str | None = None


@dataclass(frozen=True)
class DomainPack:
    """The full, validated contents of one domain's ``manifest.yaml``.

    Attributes:
        domain_id: Short machine identifier (e.g. "hospital", "banking").
        display_name: Human-readable name shown in logs/UI.
        agent_name: LiveKit explicit-dispatch name (``WorkerOptions
            (agent_name=...)``) - must be globally unique across every
            pack ``AgentAssembler`` loads; validated at load time.
        timezone: IANA timezone name (e.g. "Asia/Karachi"). Required, no
            fallback - the ``HOSPITAL_TZ``/``booking.py`` bug class (a
            container's naive system clock silently used instead of the
            domain's real local time) bit this codebase twice already.
        persona: See ``PersonaConfig``.
        engines: ``{"llm": EngineSelection, "stt": EngineSelection, "tts":
            EngineSelection}``.
        tools: See ``ToolsConfig``.
        safety: See ``SafetyConfig``.
        filler: See ``FillerConfig``.
        recap: See ``RecapConfig``.
        kb: See ``KbConfig``.
        compliance_profile: Name of a profile registered in
            ``compliance_profiles.py`` (e.g. "hipaa", "pci_dss_glba",
            "none").
        strip_thinking_channel: Whether to apply the Gemma-4
            thinking-channel leak filter - tied to the LLM model's own
            quirk, not the domain, so this is set per-pack based on which
            model the pack's ``engines.llm`` selects.
    """

    domain_id: str
    display_name: str
    agent_name: str
    timezone: str
    persona: PersonaConfig
    engines: dict[str, EngineSelection]
    tools: ToolsConfig
    safety: SafetyConfig
    filler: FillerConfig
    recap: RecapConfig
    kb: KbConfig
    compliance_profile: str
    strip_thinking_channel: bool = True

    def validate(self) -> list[str]:
        """Check every required field, collecting all problems at once.

        Returns:
            A list of human-readable problem descriptions - empty if the
            pack is valid. Mirrors ``system_config.save_config()``'s
            "collect everything, don't stop at the first" convention so a
            pack author sees every mistake in one pass.
        """
        problems: list[str] = []
        if not self.domain_id:
            problems.append("domain_id is required")
        if not self.agent_name:
            problems.append("agent_name is required (LiveKit explicit-dispatch name)")
        if not self.timezone:
            problems.append(
                "timezone is required (IANA name, e.g. 'Asia/Karachi') - no "
                "silent fallback to the container's system clock is allowed"
            )
        for slot in ("llm", "stt", "tts"):
            if slot not in self.engines:
                problems.append(f"engines.{slot} is required")
        if not self.tools.module:
            problems.append("tools.module is required")
        if not self.tools.allow_list:
            problems.append("tools.allow_list must list at least one tool name")
        if not self.safety.module:
            problems.append("safety.module is required")
        if not self.persona.template_path:
            problems.append("persona.template_path is required")
        return problems
