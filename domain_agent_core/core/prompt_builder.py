"""Jinja2 templated system-prompt builder, generalizing
``app/prompts.py``'s hardcoded f-string ``build_system_prompt()``.

Each domain pack supplies a ``persona.jinja2`` template plus a
``variables`` dict (``DomainPack.persona``). This module renders it with
those variables, the domain's real current local date (in its own
``DomainPack.timezone`` - never the container's naive system clock, the
same discipline ``app/prompts.py``'s ``HOSPITAL_TZ`` fix and
``app/hospital_core/booking.py``'s ``HOSPITAL_TZ`` fix both established
independently after a real production bug), and the active compliance
profile's ``prompt_compliance_clause`` (injected into a ``{% block
compliance %}`` section so the LLM-facing rule and the code-enforced rule
are always stated together).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import jinja2


def build_system_prompt(
    template_path: Path,
    variables: dict[str, object],
    timezone: str,
    compliance_clause: str = "",
) -> str:
    """Render a domain's persona template into a full system prompt.

    Args:
        template_path: Absolute path to the pack's ``persona.jinja2``
            file.
        variables: The pack's ``persona.variables`` from its manifest.
        timezone: IANA timezone name (``DomainPack.timezone``) - the
            template receives ``today`` and ``today_weekday`` computed in
            THIS timezone, never the process's system clock, so the LLM's
            "today"/"tomorrow" conversions are always correct for the
            caller's real locale.
        compliance_clause: The active ``CompliancePolicy.
            prompt_compliance_clause``, available to the template as
            ``compliance_clause``.

    Returns:
        The fully rendered system-prompt string.
    """
    today = datetime.now(ZoneInfo(timezone)).date()
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)
    return template.render(
        today=today.isoformat(),
        today_weekday=today.strftime("%A"),
        compliance_clause=compliance_clause,
        **variables,
    )


def build_greeting_instructions(greeting_template: str, variables: dict[str, object]) -> str:
    """Render a domain's short opening-turn instruction string.

    Args:
        greeting_template: The pack's ``persona.greeting_template`` (a
            Jinja2-renderable string, not a file).
        variables: The pack's ``persona.variables``.

    Returns:
        The rendered greeting instructions, passed to
        ``session.generate_reply(instructions=...)`` on ``on_enter()``.
    """
    return jinja2.Template(greeting_template).render(**variables)
