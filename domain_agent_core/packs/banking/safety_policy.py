"""Banking safety policy for the real HBL telephone-banking agent.

The real ``Livekit Pipeline`` system had no dedicated safety-gate module -
its only deterministic guardrail was an OUTPUT-side regex
(``asserts_block_success`` in ``src/banking.py``) checking the model never
*claimed* a card was blocked. This module instead lifts the Meridian demo
pack's INPUT-side safety-gate structure (fraud / account-takeover /
duress, via ``core.safety_gate``) and fills it with content genuinely
suited to this agent's real, narrower scope: it can verify identity and
block a card or queue a human callback - it cannot freeze an account or
investigate a dispute the way the synthetic Meridian pack's tools could.
Every escalation message below only ever promises what this agent can
actually do (verify-then-block, or hand off to a human), never a
completed action its own tools didn't perform.

``duress`` is kept as a silent-escalation category (a caller being forced
to hand over card/verification details under threat must never hear a
spoken "I'm alerting security" - that could tip off a listening
attacker), actually enabled via this pack's ``pci_dss_glba`` compliance
profile's ``silent_escalation_categories`` (see ``core/compliance_profiles.py``),
resolved by ``AgentAssembler.assemble()``.
"""

from __future__ import annotations

from domain_agent_core.core.safety_gate import (
    EmergencyRule,
    SafetyPolicy,
    compile_patterns,
)

FRAUD_MESSAGE = (
    "I understand your card may have been used without your permission. "
    "I can block it right away once we verify a couple of details "
    "together, starting with the last 4 digits of the card."
)

ACCOUNT_TAKEOVER_MESSAGE = (
    "This sounds like your account security may have been compromised. "
    "I'm connecting you with a specialist who can help secure your "
    "account right away."
)

# Deliberately reassuring, non-committal message: a duress call must
# escalate SILENTLY (see module docstring) - AgentAssembler wires
# escalate_silently=True for this kind via the pci_dss_glba compliance
# profile, so whatever message lives here is never actually spoken; kept
# non-empty only so a misconfigured deployment (compliance profile
# accidentally set to "none") still says something rather than silently
# ghosting the caller.
DURESS_MESSAGE = "I understand. I'm going to continue speaking normally with you now."

POLICY = SafetyPolicy(
    rules=(
        EmergencyRule(
            category="fraud_in_progress",
            kind="fraud",
            patterns=compile_patterns(
                [
                    r"someone (is|just) (using|used) my (card|account)",
                    r"unauthorized (charge|transaction|withdrawal)",
                    r"my card (was|got) (stolen|lost)",
                    r"i (lost|misplaced) my card",
                    r"i (didn'?t|did not) make (that|this) (charge|purchase|transaction)",
                ]
            ),
        ),
        EmergencyRule(
            category="account_takeover",
            kind="account_takeover",
            patterns=compile_patterns(
                [
                    r"my (password|pin) (was|got) changed and i didn'?t",
                    r"someone (else )?logged into my account",
                    r"i'?ve been locked out of my (account|online banking|mobile app)",
                    r"someone (else )?has access to my account",
                ]
            ),
        ),
        EmergencyRule(
            category="duress",
            kind="duress",
            patterns=compile_patterns(
                [
                    r"there'?s someone (here )?forcing me",
                    r"i'?m being (forced|told) to (give|hand over|share)",
                    r"someone is making me (do this|call you)",
                    r"i can'?t talk freely right now",
                ]
            ),
        ),
    ),
    escalation_messages={
        "fraud": FRAUD_MESSAGE,
        "account_takeover": ACCOUNT_TAKEOVER_MESSAGE,
        "duress": DURESS_MESSAGE,
    },
)

# Deliberate negatives included alongside positives, same discipline as
# the Meridian demo pack's own self-test (a naive matcher would get these
# wrong) - and matching the real system's own guardrail test style
# (Livekit Pipeline/tests/test_banking.py's request-vs-claim distinction).
TEST_CASES: list[tuple[str, bool]] = [
    ("someone is using my card right now", True),
    ("my card was stolen yesterday", True),
    ("I lost my card at the market", True),
    ("I didn't make this transaction, please help", True),
    ("my password was changed and I didn't do it", True),
    ("someone logged into my account without permission", True),
    ("I've been locked out of my mobile app", True),
    ("there's someone here forcing me to do this", True),
    ("I'm being forced to hand over my details", True),
    # deliberate negatives
    ("my card is in my wallet, I just want to block it as a precaution", False),
    ("can you block my card please", False),
    ("I want to talk to a human representative", False),
    ("what are your branch hours", False),
    ("what's the profit rate on the savings account", False),
    ("I forgot my mother's maiden name, can you remind me", False),
]


if __name__ == "__main__":
    from domain_agent_core.core.safety_gate import run_self_test

    run_self_test(POLICY, TEST_CASES)
