"""``BaseDomainAgent``: the ONE generic ``Agent`` subclass every domain
pack shares. Re-implements the generic lifecycle pieces
``RiversideReceptionist`` (``app/main.py:581-1072``) hardcodes as
hospital-specific - persona text, recap fields, safety policy, filler
phrases - by taking them all from an ``AssembledDomain`` instead.

Today's only ``Agent`` subclass hardcodes five tools as decorated BOUND
METHODS. This one instead passes ``tools=assembled.tool_callables`` (free
functions, confirmed supported by the installed ``livekit-agents``'s
``Agent.__init__``) to ``super().__init__()``, so a domain's tools live
entirely in that domain's own ``tools.py`` module, never in this file.
"""

from __future__ import annotations

import asyncio
from typing import Any

from livekit.agents import Agent, ModelSettings, StopResponse, llm
from loguru import logger

from domain_agent_core.core import audit_log, prompt_builder
from domain_agent_core.core.chat_ctx_guard import truncate_chat_ctx
from domain_agent_core.core.domain_loader import AssembledDomain
from domain_agent_core.core.safety_gate import run_safety_gate
from domain_agent_core.core.thinking_filter import ThinkingChannelStreamFilter
from domain_agent_core.core.turn_filler import TurnFiller


class BaseDomainAgent(Agent):
    """A domain-configured LiveKit voice agent.

    Every domain-specific constant (persona, recap fields, safety policy,
    filler phrases) comes from ``assembled`` rather than being hardcoded -
    this is the single piece of code every future domain pack relies on
    and never needs to touch.

    Attributes:
        assembled: The domain's fully-resolved ``AssembledDomain``.
        filler: This call's ``TurnFiller`` instance.
    """

    def __init__(self, assembled: AssembledDomain) -> None:
        """Build the agent for one call.

        Args:
            assembled: Result of ``AgentAssembler.assemble()`` for the
                active domain.
        """
        self.assembled = assembled
        compliance_clause = assembled.compliance.prompt_compliance_clause
        self._base_instructions = prompt_builder.build_system_prompt(
            assembled.persona_template_path,
            assembled.pack.persona.variables,
            assembled.pack.timezone,
            compliance_clause,
        )
        super().__init__(
            instructions=self._base_instructions,
            tools=list(assembled.tool_callables),
        )
        self.filler = TurnFiller(delay_secs=1.0, phrases=assembled.pack.filler.phrases)
        self._call_facts: dict[str, str] = {}

    # ------------------------------------------------------------- lifecycle
    async def on_enter(self) -> None:
        """Speak the domain's greeting on session start."""
        greeting = prompt_builder.build_greeting_instructions(
            self.assembled.pack.persona.greeting_template,
            self.assembled.pack.persona.variables,
        )
        self.session.generate_reply(instructions=greeting)

    # ------------------------------------------------------------------ LLM
    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ):
        """Strip a leaked thinking-channel block from streamed text before
        it reaches TTS, if this domain's engine needs it (see
        ``DomainPack.strip_thinking_channel``). Guarantees the caller
        always hears something even if a leak swallows the whole
        completion."""
        stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        if asyncio.iscoroutine(stream):
            stream = await stream
        if stream is None:
            return
        if not self.assembled.pack.strip_thinking_channel:
            async for chunk in stream:
                yield chunk
            return

        leak_filter = ThinkingChannelStreamFilter()
        yielded_anything = False
        async for chunk in stream:
            if isinstance(chunk, str):
                visible = leak_filter.push(chunk)
                if visible:
                    yielded_anything = True
                    yield visible
                continue
            if chunk.delta is not None and chunk.delta.content:
                visible = leak_filter.push(chunk.delta.content)
                if not visible and not chunk.delta.tool_calls:
                    continue
                chunk.delta.content = visible
            if chunk.delta is not None and (chunk.delta.content or chunk.delta.tool_calls):
                yielded_anything = True
            yield chunk
        if not yielded_anything:
            logger.error(
                "llm_node: entire completion swallowed by the thinking-channel "
                "leak filter - speaking a fallback so the caller always "
                "hears something."
            )
            yield "Sorry, could you say that again?"

    # ------------------------------------------------------------ TURN FILLER
    def cancel_pending_filler(self) -> None:
        """Called on every "speaking" transition - see ``TurnFiller.
        cancel()``."""
        self.filler.cancel()

    # -------------------------------------------------------- CALL-FACT RECAP
    async def remember(self, **facts: str | None) -> None:
        """Fold whatever the caller just stated into the agent's
        instructions, keyed by this domain's ``DomainPack.recap.labels``.

        Survives ``CHAT_CTX_MAX_ITEMS`` truncation - the point is that
        truncation only guarantees the *first* instructions message
        survives, not any particular chat turn, so a fact stated early in
        a long call would otherwise be silently forgotten.

        Args:
            **facts: Keyword args matching this domain's recap label
                keys (e.g. ``patient_name=...`` for hospital,
                ``account_last4=...`` for banking). ``None``/omitted
                values are ignored.
        """
        changed = False
        for key, _label in self.assembled.pack.recap.labels:
            value = facts.get(key)
            if value and self._call_facts.get(key) != value:
                self._call_facts[key] = value
                changed = True
        if not changed:
            return

        recap_parts = [
            f"{label}: {self._call_facts[key]}"
            for key, label in self.assembled.pack.recap.labels
            if key in self._call_facts
        ]
        recap = "Known so far this call (do not ask again for these): " + "; ".join(
            recap_parts
        )
        await self.update_instructions(f"{self._base_instructions}\n\n{recap}")

    # ------------------------------------------------------------ SAFETY GATE
    async def on_user_turn_completed(self, turn_ctx: Any, new_message: Any) -> None:
        """Deterministic emergency bypass, run BEFORE any LLM inference
        for this turn. On a match: speak (or silently log, per
        ``escalate_silently``) the escalation message and ``StopResponse()``
        so no generation happens."""
        truncate_chat_ctx(turn_ctx)
        text = (getattr(new_message, "text_content", None) or "").strip()
        if not text:
            return

        hit = run_safety_gate(text, self.assembled.safety_policy)
        if hit is None:
            self.cancel_pending_filler()
            self.filler.arm(self.session)
            return

        logger.warning(
            "SAFETY GATE: emergency matched "
            f"(category={hit['category']}, kind={hit['kind']}, "
            f"matched_text={hit['matched_text']!r}, "
            f"silent={hit['escalate_silently']}) - bypassing LLM"
        )
        audit_log.record(
            self.assembled.audit_dir,
            self.assembled.pack.domain_id,
            self.assembled.compliance,
            event_type="safety_escalation",
            payload={
                "category": hit["category"],
                "kind": hit["kind"],
                "escalate_silently": hit["escalate_silently"],
            },
        )
        if not hit["escalate_silently"]:
            try:
                self.session.say(hit["message"], add_to_chat_ctx=False)
            except TypeError:
                self.session.say(hit["message"])
        raise StopResponse()

    # ------------------------------------------------------------------ tools
    async def record_tool_call(self, tool_name: str, args: dict, result: dict) -> None:
        """Route a tool call through compliance-aware audit logging.

        Args:
            tool_name: The tool's function name.
            args: The kwargs the LLM passed.
            result: The tool's return value.
        """
        audit_log.record(
            self.assembled.audit_dir,
            self.assembled.pack.domain_id,
            self.assembled.compliance,
            event_type="tool_call",
            payload={"tool": tool_name, "args": args, "result": result},
        )
