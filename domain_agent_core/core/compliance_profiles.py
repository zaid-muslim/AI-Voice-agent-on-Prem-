"""Compliance profiles: what each vertical's regulatory obligations
concretely parameterize in code, not just in the persona prompt.

Healthcare (HIPAA) and banking (GLBA/PCI-DSS/FINRA) differ meaningfully in
what they require - audit-log retention length, whether raw account/card
numbers may ever be logged, and whether an agent may assert a fact the LLM
"recalled" versus only ever a tool's literal return value. Each profile
below is a frozen, named bundle of those concrete knobs, referenced by
name from a pack's ``manifest.yaml`` (``compliance.profile``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompliancePolicy:
    """Concrete, enforceable compliance knobs for one profile.

    Attributes:
        name: Profile identifier (e.g. "hipaa", "pci_dss_glba", "none").
        audit_log_level: "none" (no audit writes beyond normal logging),
            "standard" (log tool calls), or "strict" (log tool calls with
            redaction applied and never rotated away).
        retention_days: Minimum days audit files must be kept before
            they're eligible for external archival. Never used to delete
            anything from this codebase - see ``audit_log.py``.
        redact_fields: Field names that must be hashed (not merely
            masked) before ever reaching a log line or a ``push_ui``
            payload.
        redact_patterns: Compiled regexes scrubbed from free-text fields
            (transcripts, tool-call args) before logging - a defense-in-
            depth net, not the primary control.
        require_deterministic_tool_backing: If True, ``tool_registry.py``
            enforces that every allow-listed tool is backed by real I/O
            (asserted via a ``DETERMINISTIC`` name-set the pack's tools
            module must export), not something the LLM could assert on
            its own initiative.
        prompt_compliance_clause: Text injected into the persona
            template's ``{% block compliance %}`` section, so the
            LLM-facing rule and the code-enforced rule are stated
            together.
        silent_escalation_categories: Safety-gate categories that must
            escalate WITHOUT speaking anything to the caller (e.g. a
            banking duress call, where a spoken "I'm alerting security"
            could tip off a listening attacker) - still logged, just not
            said aloud.
    """

    name: str
    audit_log_level: str
    retention_days: int
    redact_fields: frozenset[str] = field(default_factory=frozenset)
    redact_patterns: tuple[re.Pattern, ...] = field(default_factory=tuple)
    require_deterministic_tool_backing: bool = False
    prompt_compliance_clause: str = ""
    silent_escalation_categories: frozenset[str] = field(default_factory=frozenset)


_PII_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-shaped
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # date-of-birth-shaped, YYYY-MM-DD
)

_ACCOUNT_NUMBER_PATTERNS = (
    re.compile(r"\b\d{12,19}\b"),  # card/account-number-shaped runs of digits
)

_PROFILES: dict[str, CompliancePolicy] = {
    "none": CompliancePolicy(
        name="none",
        audit_log_level="standard",
        retention_days=0,
        require_deterministic_tool_backing=False,
        prompt_compliance_clause="",
    ),
    "hipaa": CompliancePolicy(
        name="hipaa",
        audit_log_level="strict",
        retention_days=2190,  # 6 years - HHS's HIPAA minimum retention
        redact_fields=frozenset(),  # identity IS the auditable record here, not anonymized
        redact_patterns=_PII_PATTERNS,
        require_deterministic_tool_backing=True,
        prompt_compliance_clause=(
            "COMPLIANCE: This is a HIPAA-covered conversation. Apply the "
            "minimum-necessary standard - never volunteer more of a "
            "caller's health information than the current request needs, "
            "and never state a clinical fact you were not given by a tool "
            "result."
        ),
    ),
    "pci_dss_glba": CompliancePolicy(
        name="pci_dss_glba",
        audit_log_level="strict",
        retention_days=2555,  # 7 years - common FINRA/SEC recordkeeping baseline
        redact_fields=frozenset({"account_number", "card_number", "ssn"}),
        redact_patterns=_PII_PATTERNS + _ACCOUNT_NUMBER_PATTERNS,
        require_deterministic_tool_backing=True,
        prompt_compliance_clause=(
            "COMPLIANCE: This is a regulated financial conversation. Never "
            "state a balance, transfer status, or transaction outcome that "
            "did not come directly from a tool result. Never read a full "
            "account or card number aloud - use only the last 4 digits. "
            "Escalate suspected fraud or disputes above routine amounts "
            "rather than resolving them conversationally."
        ),
        silent_escalation_categories=frozenset({"duress"}),
    ),
}


def get_compliance_policy(name: str) -> CompliancePolicy:
    """Resolve a compliance-profile name to its ``CompliancePolicy``.

    Args:
        name: Profile name from a pack's ``manifest.yaml`` (``compliance.
            profile``), e.g. "hipaa".

    Returns:
        The registered ``CompliancePolicy`` for that name.

    Raises:
        KeyError: If ``name`` isn't a registered profile - listing the
            valid names in the message so a manifest typo is obvious.
    """
    try:
        return _PROFILES[name]
    except KeyError:
        raise KeyError(
            f"unknown compliance profile {name!r} - valid profiles: "
            f"{sorted(_PROFILES)}"
        ) from None
