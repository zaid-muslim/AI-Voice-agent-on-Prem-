"""Banking safety policy - the second proof point for the safety-gate
engine: same ``core.safety_gate`` matching logic as the hospital pack,
completely different rule content.

``duress`` is the genuinely new requirement this domain surfaces that
hospital never needed: a caller being forced to withdraw/transfer under
threat must never hear a spoken "I'm alerting security" - that could tip
off a listening attacker. This pack marks ``duress`` as a candidate for
``escalate_silently`` (actually enabled via the pack's ``pci_dss_glba``
compliance profile's ``silent_escalation_categories``, resolved by
``AgentAssembler.assemble()`` - see ``core/compliance_profiles.py``).
"""

from __future__ import annotations

from domain_agent_core.core.safety_gate import (
    EmergencyRule,
    SafetyPolicy,
    compile_patterns,
)

FRAUD_HOTLINE_MESSAGE = (
    "I've flagged this as suspected fraud and frozen the account right "
    "away. A fraud specialist will call you back shortly. If you notice "
    "any other unauthorized activity, please call our fraud hotline "
    "immediately."
)

ACCOUNT_TAKEOVER_MESSAGE = (
    "This sounds like your account security may have been compromised. "
    "I'm flagging this for our security team right now, and I'd recommend "
    "changing your online banking password as soon as we're done here."
)

# Deliberately empty spoken message: a duress call must escalate SILENTLY
# (see module docstring) - AgentAssembler wires escalate_silently=True for
# this kind via the pci_dss_glba compliance profile, so whatever message
# lives here is never actually spoken; kept non-empty only so a
# misconfigured deployment (compliance profile accidentally set to
# "none") still says something rather than silently ghosting the caller.
DURESS_MESSAGE = (
    "I understand. I'm going to continue speaking normally with you now."
)

POLICY = SafetyPolicy(
    rules=(
        EmergencyRule(
            category="fraud_in_progress",
            kind="fraud",
            patterns=compile_patterns(
                [
                    r"someone (is|just) (using|used) my (card|account)",
                    r"unauthorized (charge|transaction|withdrawal)",
                    r"my card (was|got) stolen",
                    r"i (didn'?t|did not) make (that|this) (charge|purchase|transaction)",
                ]
            ),
        ),
        EmergencyRule(
            category="account_takeover",
            kind="account_takeover",
            patterns=compile_patterns(
                [
                    r"i (didn'?t|did not) (make|request) this (transfer|change)",
                    r"my (password|pin) (was|got) changed and i didn'?t",
                    r"someone (else )?logged into my account",
                    r"i'?ve been locked out of my (account|online banking)",
                ]
            ),
        ),
        EmergencyRule(
            category="duress",
            kind="duress",
            patterns=compile_patterns(
                [
                    r"there'?s someone (here )?forcing me",
                    r"i'?m being (forced|told) to (withdraw|transfer)",
                    r"someone is making me (do this|call you)",
                    r"i can'?t talk freely right now",
                ]
            ),
        ),
    ),
    escalation_messages={
        "fraud": FRAUD_HOTLINE_MESSAGE,
        "account_takeover": ACCOUNT_TAKEOVER_MESSAGE,
        "duress": DURESS_MESSAGE,
    },
)

# Deliberate negatives included alongside positives, same discipline as
# hospital_core/safety.py's own self-test (a naive matcher would get
# these wrong).
TEST_CASES: list[tuple[str, bool]] = [
    ("someone is using my card right now", True),
    ("there's an unauthorized charge on my statement", True),
    ("my card was stolen yesterday", True),
    ("I didn't make this transfer, please help", True),
    ("my password was changed and I didn't do it", True),
    ("someone logged into my account without permission", True),
    ("there's someone here forcing me to withdraw cash", True),
    ("I'm being told to transfer money right now", True),
    # deliberate negatives
    ("my card is in my wallet, I just want to check the balance", False),
    ("I want to transfer money to my savings account", False),
    ("can you help me dispute an old transaction", False),
    ("I forgot my password, can you help me reset it", False),
    ("what are your branch hours", False),
    ("I'd like to report that I'm moving to a new address", False),
]


if __name__ == "__main__":
    from domain_agent_core.core.safety_gate import run_self_test

    run_self_test(POLICY, TEST_CASES)
