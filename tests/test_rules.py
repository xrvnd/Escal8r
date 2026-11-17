import pytest
from conftest import jev_decision

from escal8r.config import Thresholds
from escal8r.jev import ALWAYS_HUMAN_GATES
from escal8r.rules import pick_action, reply_check_failure

T = Thresholds()


def test_clear_faq_question_is_a_reply():
    assert pick_action(jev_decision(), T, has_requests=True).action == "reply"


@pytest.mark.parametrize("gate", ALWAYS_HUMAN_GATES)
def test_each_always_human_gate_hands_off(gate):
    outcome = pick_action(jev_decision(**{gate: 0.40}), T, has_requests=True)
    assert (outcome.action, outcome.rule) == ("handoff", "always_human")


def test_always_human_gate_is_checked_before_spam():
    outcome = pick_action(
        jev_decision(spam=0.99, refund_or_disputed_charge=0.5), T, has_requests=True
    )
    assert outcome.action == "handoff"


def test_clear_spam_closes():
    assert pick_action(jev_decision(spam=0.9), T, has_requests=False).action == "close"


def test_borderline_spam_does_not_close():
    outcome = pick_action(jev_decision(spam=0.8, answerable=0.1), T, has_requests=True)
    assert outcome.action == "handoff"


def test_low_choice_confidence_hands_off():
    outcome = pick_action(jev_decision(confidence=0.6), T, has_requests=True)
    assert outcome.rule == "low_confidence"


@pytest.mark.parametrize("priority", ["P1", "P2"])
def test_urgent_priority_hands_off(priority):
    outcome = pick_action(jev_decision(priority=priority), T, has_requests=True)
    assert outcome.rule == "urgent"


@pytest.mark.parametrize("gate", ["answerable", "covered"])
def test_not_answerable_hands_off(gate):
    outcome = pick_action(jev_decision(**{gate: 0.5}), T, has_requests=True)
    assert outcome.rule == "not_in_faq"


def test_no_verified_requests_hands_off():
    assert pick_action(jev_decision(), T, has_requests=False).rule == "no_requests"


def test_injection_alone_does_not_force_handoff():
    outcome = pick_action(jev_decision(injection_attempt=0.99), T, has_requests=True)
    assert outcome.action == "reply"


def test_threshold_edges():
    at_gate = pick_action(jev_decision(legal_threat=T.always_human), T, has_requests=True)
    below = pick_action(jev_decision(legal_threat=T.always_human - 0.01), T, has_requests=True)
    assert at_gate.action == "handoff" and below.action == "reply"


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
