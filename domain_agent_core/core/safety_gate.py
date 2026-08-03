"""Generic two-layer safety-gate engine, generalized from
``app/hospital_core/safety.py``.

DESIGN (unchanged from the original, see that file for the full
rationale): pattern matching, not an LLM call - near-instant, deterministic,
and can't be talked out of firing by clever phrasing. Patterns match
PHRASES, not risky single words, to avoid false positives ("killing time"
must never trigger on "kill"). This is layer one of two - the persona
template's ``{% block safety %}``/compliance clause is the second, slower
layer for ambiguous cases this regex layer misses (see
``hospital_core/safety.py``'s own docstring, which names this two-layer
design explicitly).

WHAT GENERALIZED: ``RULES``/``ESCALATION_MESSAGE`` are no longer module
globals - every domain pack builds its own ``SafetyPolicy`` instance with
its own rules and messages, instantiated in that pack's own
``safety_policy.py``. The matching engine (``check_emergency``/
``run_safety_gate``) is unchanged logic. The self-test harness shape
(a list of ``(text, expected_bool)`` cases + a pass/fail printer) is kept
as a reusable ``run_self_test()`` helper so every pack's safety policy gets
the identical regression harness for free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmergencyRule:
    """One category of emergency trigger phrases.

    Attributes:
        category: Short label, e.g. "cardiac" or "fraud_in_progress".
        kind: Groups rules that share an escalation message/behavior
            (e.g. "medical" vs "self_harm" for hospital, "fraud" vs
            "duress" for banking).
        patterns: Compiled, case-insensitive regexes, matched with
            ``re.search`` against the caller's transcript.
    """

    category: str
    kind: str
    patterns: tuple[re.Pattern, ...]


def compile_patterns(patterns: list[str]) -> tuple[re.Pattern, ...]:
    """Compile a list of regex strings, case-insensitive.

    Args:
        patterns: Raw regex strings.

    Returns:
        Compiled ``re.Pattern`` objects, same order.
    """
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


@dataclass(frozen=True)
class EmergencyMatch:
    """A single safety-gate hit.

    Attributes:
        category: The matching rule's ``category``.
        kind: The matching rule's ``kind``.
        matched_text: The exact substring that matched.
    """

    category: str
    kind: str
    matched_text: str


@dataclass(frozen=True)
class SafetyPolicy:
    """One domain's complete safety-gate content - rules plus messages.

    Each domain pack's ``safety_policy.py`` builds one of these as its
    module-level ``POLICY``. The matching engine below (``check_emergency``
    / ``run_safety_gate``) is generic; only this data is domain-specific.

    Attributes:
        rules: Every ``EmergencyRule`` this domain checks, in priority
            order (first match wins).
        escalation_messages: ``kind`` -> message spoken to the caller on a
            match of that kind.
        silent_kinds: ``kind`` values that must escalate WITHOUT speaking
            anything (see ``compliance_profiles.CompliancePolicy.
            silent_escalation_categories`` - typically populated from
            there when the pack is assembled).
    """

    rules: tuple[EmergencyRule, ...]
    escalation_messages: dict[str, str] = field(default_factory=dict)
    silent_kinds: frozenset[str] = field(default_factory=frozenset)


def check_emergency(text: str, policy: SafetyPolicy) -> EmergencyMatch | None:
    """Fast, deterministic check against one domain's rules.

    Args:
        text: The caller's raw transcript for this turn.
        policy: The active domain's ``SafetyPolicy``.

    Returns:
        ``None`` if nothing matched (safe to proceed normally), or an
        ``EmergencyMatch`` if this turn should be escalated instead.
    """
    for rule in policy.rules:
        for pattern in rule.patterns:
            match = pattern.search(text)
            if match:
                return EmergencyMatch(rule.category, rule.kind, match.group(0))
    return None


def run_safety_gate(text: str, policy: SafetyPolicy) -> dict | None:
    """Run the safety gate on a raw transcript for one domain.

    Args:
        text: The caller's raw transcript for this turn.
        policy: The active domain's ``SafetyPolicy``.

    Returns:
        ``None`` if the turn should proceed normally, or a dict with
        ``emergency``, ``category``, ``kind``, ``matched_text``,
        ``message``, and ``escalate_silently`` keys if it should be
        escalated instead (bypassing the LLM entirely).
    """
    match = check_emergency(text, policy)
    if match is None:
        return None
    return {
        "emergency": True,
        "category": match.category,
        "kind": match.kind,
        "matched_text": match.matched_text,
        "message": policy.escalation_messages.get(match.kind, ""),
        "escalate_silently": match.kind in policy.silent_kinds,
    }


def run_self_test(
    policy: SafetyPolicy, test_cases: list[tuple[str, bool]], *, verbose: bool = True
) -> bool:
    """Regression harness for a domain's safety policy.

    Mirrors ``hospital_core/safety.py``'s original ``_run_self_test()``
    shape exactly, generalized to take any policy/case list so every pack
    gets the same harness for free.

    Args:
        policy: The ``SafetyPolicy`` to test.
        test_cases: ``(text, should_trigger)`` pairs - include deliberate
            negatives (phrasing a naive matcher would get wrong), not just
            positives.
        verbose: If True, print one PASS/FAIL line per case plus a
            summary, matching the original CLI-driven self-test's output.

    Returns:
        True if every case passed.
    """
    passed = 0
    for text, expected in test_cases:
        match = check_emergency(text, policy)
        got = match is not None
        ok = got == expected
        if ok:
            passed += 1
        if verbose:
            status = "PASS" if ok else "FAIL"
            detail = f"-> {match.category}" if match else ""
            print(
                f"[{status}] expected={expected!s:5} got={got!s:5} "
                f"{detail:20} | {text!r}"
            )
    if verbose:
        print(f"\n{passed}/{len(test_cases)} passed")
        if passed != len(test_cases):
            print("DO NOT wire this policy into the pipeline until all cases pass.")
    return passed == len(test_cases)
