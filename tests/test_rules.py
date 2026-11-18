import pytest
from conftest import jev_decision

from escal8r.config import Thresholds
from escal8r.jev import ALWAYS_HUMAN_GATES
from escal8r.rules import compute_priority, pick_action, reply_check_failure, reply_sections

T = Thresholds()


def act(priority="P4", has_requests=True, **kwargs):
    return pick_action(jev_decision(**kwargs), T, has_requests=has_requests, priority=priority)


# -- action -------------------------------------------------------------------------


def test_clear_faq_question_is_a_reply():
    outcome = act()
    assert (outcome.action, outcome.route) == ("reply", "faq_team_and_rota")


@pytest.mark.parametrize("gate", ALWAYS_HUMAN_GATES)
def test_each_always_human_gate_hands_off_with_a_queue(gate):
    outcome = act(**{gate: 0.40})
    assert (outcome.action, outcome.rule) == ("handoff", "always_human")
    assert outcome.queue


def test_always_human_gate_beats_spam():
    outcome = act(spam=0.99, refund_or_disputed_charge=0.5, route_probabilities={"spam": 0.95})
    assert outcome.action == "handoff"


def test_spam_closes_only_when_gate_and_route_agree():
    assert act(spam=0.9, route_probabilities={"spam": 0.9}).action == "close"
    disagree = act(spam=0.9, route_probabilities={"faq_redirect": 0.9})
    assert (disagree.action, disagree.rule) == ("handoff", "spam_unsure")


def test_low_choice_confidence_hands_off():
    assert act(confidence=0.6).rule == "low_confidence"


@pytest.mark.parametrize("priority", ["P1", "P2"])
def test_urgent_priority_hands_off(priority):
    assert act(priority=priority).rule == "urgent"


def test_grouped_human_mass_hands_off_even_if_top_route_is_faq():
    outcome = act(
        route_probabilities={
            "faq_plans_invoices": 0.40,
            "human_billing_dispute": 0.20,
            "human_engineering": 0.20,
            "faq_redirect": 0.20,
        }
    )
    assert (outcome.rule, outcome.route, outcome.queue) == (
        "human_route",
        "human_billing_dispute",
        "Billing",
    )


def test_small_margin_between_faq_and_human_hands_off():
    outcome = act(
        route_probabilities={
            "faq_team_and_rota": 0.30,
            "human_engineering": 0.20,
            "faq_redirect": 0.5,
        }
    )
    assert outcome.action == "reply"  # margin 0.5 - 0.2 = 0.3
    torn = act(
        route_probabilities={
            "faq_team_and_rota": 0.38,
            "human_engineering": 0.30,
            "faq_redirect": 0.32,
        }
    )
    assert torn.rule == "ambiguous_route"  # margin 0.38 - 0.30 = 0.08


@pytest.mark.parametrize("gate", ["answerable", "covered"])
def test_not_answerable_hands_off(gate):
    assert act(**{gate: 0.5}).rule == "not_in_faq"


def test_no_verified_requests_hands_off():
    assert act(has_requests=False).rule == "no_requests"


def test_injection_alone_does_not_force_handoff():
    assert act(injection_attempt=0.99).action == "reply"


def test_threshold_edges():
    assert act(legal_threat=T.always_human).action == "handoff"
    assert act(legal_threat=T.always_human - 0.01).action == "reply"


# -- priority: the brief's table ------------------------------------------------------


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        ({"security_incident": 0.9}, "P1"),
        ({"legal_threat": 0.9}, "P1"),
        ({"users_affected": 2.0, "blocked_now": 0.9}, "P1"),
        ({"refund_or_disputed_charge": 0.9}, "P2"),
        ({"cancellation_or_contract_change": 0.9}, "P2"),
        ({"churn_threat": 0.9}, "P2"),
        ({"personal_data_request": 0.9}, "P2"),
        ({"blocked_now": 0.9}, "P2"),  # one customer, no workaround
        ({"category": "bug", "deadline_soon": 0.9}, "P2"),
        ({"category": "bug", "cosmetic_only": 0.9}, "P4"),
        ({"category": "bug"}, "P3"),
        ({"blocked_now": 0.9, "workaround_exists": 0.9}, "P3"),
        ({"users_affected": 1.0, "blocked_now": 0.9, "workaround_exists": 0.9}, "P3"),
        ({}, "P4"),
    ],
)
def test_priority_table(facts, expected):
    priority, why = compute_priority(jev_decision(**facts), T)
    assert priority == expected, why


def test_many_users_needs_to_be_blocked_for_p1():
    priority, _ = compute_priority(jev_decision(users_affected=3.0, category="bug"), T)
    assert priority == "P3"


# -- reply sections and verification ------------------------------------------------------


def test_reply_sections():
    assert reply_sections("faq_redirect", []) == ["F10", "F11"]  # no citation: route decides
    assert reply_sections("faq_redirect", ["F10", "F02"]) == ["F10"]  # overlap narrows
    assert reply_sections("faq_redirect", ["F02"]) is None  # disagreement


@pytest.mark.parametrize(
    ("scores", "fails"),
    [
        ({"grounded": 0.9, "reply_language": 0.9, "commitment": 0.1}, False),
        ({"grounded": 0.7, "reply_language": 0.9, "commitment": 0.1}, True),
        ({"grounded": 0.9, "reply_language": 0.5, "commitment": 0.1}, True),
        ({"grounded": 0.9, "reply_language": 0.9, "commitment": 0.5}, True),
    ],
)
def test_reply_check(scores, fails):
    assert (reply_check_failure(scores, T) is not None) == fails
