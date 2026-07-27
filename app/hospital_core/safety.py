"""
Safety gate: runs on the raw transcript, BEFORE the LLM orchestrator, RAG,
or any tool call. If it matches, everything else is bypassed and a fixed
escalation message is spoken instead.

DESIGN CHOICES (read before wiring this in):

1. Pattern matching, not an LLM call. This must be near-instant (microseconds)
   and deterministic - it cannot wait on a model, and it cannot be talked out
   of firing by clever phrasing the way a prompt-based check sometimes can.
   Cost: it will miss phrasings not in these lists. That's why this is
   "defense in depth", not the only layer - see note #3.

2. Patterns match PHRASES, not risky single words. "kill" alone would false-
   positive on "I'm killing time waiting for my appointment". Patterns like
   "kill myself" / "going to kill myself" avoid that. Every pattern here was
   chosen to be as specific as reasonably possible while still catching real
   distress. Run the self-test at the bottom before trusting this on real
   (or real-feeling test) calls - it includes deliberate negative cases.

3. THIS IS NOT SUFFICIENT ALONE. The LLM orchestrator's system prompt should
   ALSO be instructed to treat ambiguous distress gently and escalate or ask
   a clarifying safety question rather than assume everything not matched
   here is fine. This module catches the clear cases fast; the LLM is the
   second, slower layer for everything in between.

4. EMERGENCY_NUMBER below is a placeholder. Set it to the correct local
   emergency number for wherever this is actually deployed - do not ship
   "911" if the deployment region uses a different number.

5. This is a heuristic prototype for a demo/test hospital agent, not a
   validated clinical triage system. If this were ever used for real
   patients, it would need review by clinical/safety staff, much broader
   phrase coverage, and likely a second ML-based check - not just this file.

LIVEKIT COMPAT NOTE: unchanged from the original. run_safety_gate(text) ->
Optional[dict] never used Pipecat's params/callback pattern, so it already
matches compat.py's (and a LiveKit agent's) plain args-in/dict-out shape.
"""

import re
from dataclasses import dataclass
from typing import Optional

# --- CONFIG: set for your deployment ---------------------------------------

EMERGENCY_NUMBER = "1122"  # Pakistan's nationwide Rescue service (confirmed
# operational in Islamabad via CARES 1122). Update this again if this agent
# is ever deployed somewhere else - always verify the real local number,
# don't assume "911" or any other number is universal.

# --- rules -------------------------------------------------------------


@dataclass
class EmergencyRule:
    category: str  # short label, e.g. "cardiac"
    kind: str  # "medical" or "self_harm" - controls which message fires
    patterns: list  # compiled regexes, matched with re.search, case-insensitive


def _compile(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


RULES = [
    EmergencyRule(
        "cardiac",
        "medical",
        _compile(
            [
                r"chest pain",
                r"heart attack",
                r"crushing (pain|pressure) in my chest",
                r"pain (radiating|spreading) (down|to) my (arm|jaw)",
            ]
        ),
    ),
    EmergencyRule(
        "breathing",
        "medical",
        _compile(
            [
                r"can\W?t breathe",
                r"cannot breathe",
                r"(having trouble|struggling) breathing",
                r"\bchoking\b",
            ]
        ),
    ),
    EmergencyRule(
        "stroke",
        "medical",
        _compile(
            [
                r"face is drooping",
                r"slurred speech",
                r"can\W?t speak (properly|right)",
                r"sudden numbness",
                r"worst headache of my life",
                r"can\W?t move (my |one )?(arm|leg|side)",
            ]
        ),
    ),
    EmergencyRule(
        "trauma_bleeding",
        "medical",
        _compile(
            [
                r"severe bleeding",
                r"bleeding (heavily|a lot|and (it|won\W?t) (won\W?t stop|stop))",
                r"been (shot|stabbed)",
                r"\bunconscious\b",
                r"not breathing",
                r"\bunresponsive\b",
            ]
        ),
    ),
    EmergencyRule(
        "overdose",
        "medical",
        _compile(
            [
                r"took too many (pills|tablets)",
                r"\boverdose(d)?\b",
                r"swallowed (a bunch of|too many) (pills|medication)",
            ]
        ),
    ),
    EmergencyRule(
        "self_harm",
        "self_harm",
        _compile(
            [
                r"(want|going|planning) to kill myself",
                r"end my life",
                r"\bsuicidal\b",
                r"(going to |want to )?hurt myself",
                r"harm myself",
                r"don\W?t want to (live|be alive) (anymore|any more)",
            ]
        ),
    ),
]

ESCALATION_MESSAGE = {
    "medical": (
        f"This sounds like it could be a medical emergency. Please hang up "
        f"right now and call {EMERGENCY_NUMBER}, or go to your nearest "
        f"emergency room immediately."
    ),
    "self_harm": (
        "I'm concerned about your safety, and I want to make sure you get "
        "real help right now. If you're in immediate danger, please call "
        f"{EMERGENCY_NUMBER} or go to your nearest emergency room. You can "
        "also reach a crisis line for support - you don't have to go "
        "through this alone."
    ),
}


@dataclass
class EmergencyMatch:
    category: str
    kind: str
    matched_text: str


def check_emergency(text: str) -> Optional[EmergencyMatch]:
    """Fast, deterministic check. Returns None if nothing matched (safe to
    proceed to the normal orchestrator), or an EmergencyMatch if it should be
    escalated instead."""
    for rule in RULES:
        for pattern in rule.patterns:
            m = pattern.search(text)
            if m:
                return EmergencyMatch(rule.category, rule.kind, m.group(0))
    return None


def run_safety_gate(text: str) -> Optional[dict]:
    """Call this first, on the raw transcript. Returns None if the turn
    should proceed normally, or a dict with the escalation message to speak
    instead (bypassing the LLM orchestrator entirely) if not."""
    match = check_emergency(text)
    if match is None:
        return None
    return {
        "emergency": True,
        "category": match.category,
        "kind": match.kind,
        "matched_text": match.matched_text,
        "message": ESCALATION_MESSAGE[match.kind],
    }


# --- self-test: run this file directly before wiring it into the pipeline --

_TEST_CASES = [
    # (text, should_trigger)
    ("I'm having really bad chest pain right now", True),
    ("I think I'm having a heart attack", True),
    ("I can't breathe, please help", True),
    ("my face is drooping and my speech is slurred", True),
    ("this is the worst headache of my life", True),
    ("there's severe bleeding and it won't stop", True),
    ("I took too many pills an hour ago", True),
    ("I want to kill myself", True),
    ("I don't want to be alive anymore", True),
    # deliberate negatives - things a naive matcher would get wrong
    ("I'm killing time waiting for my appointment", False),
    ("I have a small headache today", False),
    ("can you help me book an appointment with a cardiologist", False),
    ("my chest has been a bit sore after the gym", False),
    ("I'd like to check my medication refill status", False),
    ("what are your visiting hours", False),
]


def _run_self_test():
    passed = 0
    for text, expected in _TEST_CASES:
        match = check_emergency(text)
        got = match is not None
        status = "PASS" if got == expected else "FAIL"
        if status == "PASS":
            passed += 1
        detail = f"-> {match.category}" if match else ""
        print(
            f"[{status}] expected={expected!s:5} got={got!s:5} {detail:20} | {text!r}"
        )
    print(f"\n{passed}/{len(_TEST_CASES)} passed")
    if passed != len(_TEST_CASES):
        print("DO NOT wire this into the pipeline until all cases pass.")


if __name__ == "__main__":
    _run_self_test()
