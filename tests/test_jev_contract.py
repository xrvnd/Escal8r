"""Contract test against the REAL alpha Decisions endpoint. Costs a fraction of a cent.

Deselected by default. Run it on purpose:  uv run pytest -m network
It fails loudly if the endpoint's request or response shape changes.
"""

from __future__ import annotations

import pytest

from escal8r.config import load_settings
from escal8r.jev import ALWAYS_HUMAN_GATES, Jev
from escal8r.openrouter import CostTracker, OpenRouterClient

SYNTHETIC_TICKET = {
    "ticket": {
        "subject": "Refund please",
        "body": "We were closed in August. Can we get that month refunded?",
        "customer_plan": "Basis",
        "channel": "email",
    },
    "interpreter_evidence": "unavailable: read the ticket directly",
    "help_centre_faq": "F05 · Plans and prices\nBasis: €4 per active employee per month.",
    "support_team_rules": "Refunds always go to a human.",
}


@pytest.mark.network
def test_decisions_endpoint_contract():
    settings = load_settings()
    if not settings.openrouter_key:
        pytest.skip("OPENROUTER_KEY not set")
    tracker = CostTracker(limit_usd=0.05)
    with OpenRouterClient(settings.openrouter_key, tracker) as client:
        decision = Jev(client).decide(SYNTHETIC_TICKET)

    assert decision.language_code == "en"
    assert set(ALWAYS_HUMAN_GATES) <= set(decision.gates)
    assert decision.gates["refund_or_disputed_charge"] > 0.5
    call = tracker.calls[0]
    print(
        f"\nserved by {call.model_served}; cost reported: {call.cost_reported} "
        f"(${call.cost_usd:.6f}); tokens in/out: {call.input_tokens}/{call.output_tokens}"
    )
