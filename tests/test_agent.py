"""The per-ticket pipeline with fake models: every path, no network."""

from __future__ import annotations

import json

import pytest
from conftest import jev_decision

from escal8r.agent import Agent
from escal8r.config import FAQ_PATH, Thresholds
from escal8r.deepseek import Interpretation, Request
from escal8r.faq import load_faq
from escal8r.models import Ticket
from escal8r.openrouter import CallRecord, CostTracker, ModelError

FAQ = load_faq(FAQ_PATH)
HOW_TO = Ticket(
    id="T005",
    customer_plan="Basis",
    subject="Nieuwe medewerker",
    body="Hoe voeg ik een nieuwe medewerker toe aan ons rooster? Groet, Anouk",
)
GOOD_SCORES = {"grounded": 0.95, "reply_language": 0.97, "commitment": 0.03}


class FakeDeepSeek:
    def __init__(self, tracker: CostTracker, *, fail: set[str] | None = None, draft: str = ""):
        self.tracker = tracker
        self.fail = fail or set()
        self.draft = draft or "Hoi, ga naar Team → Medewerker toevoegen. Team Roosterly"
        self.seen: list[str] = []
        self.called: list[str] = []

    def _call(self, name: str) -> None:
        self.called.append(name)
        self.tracker.calls.append(CallRecord(name, "ds", "ds-served", 1000, 100, 0.0001, True))
        if name in self.fail:
            raise ModelError(f"{name} down")

    def interpret(self, ticket: Ticket) -> tuple[Interpretation, int]:
        self.seen.append(ticket.body)
        self._call("interpret")
        requests = [Request(text_en="How to add an employee", quote="Hoe voeg ik")]
        return (
            Interpretation(
                language_guess="de",
                english_summary="Wants to add an employee.",
                requests=requests,
                faq_ids=["F02"],
            ),
            1,
        )

    def write_reply(self, **kwargs: object) -> tuple[str, str]:
        self._call("write_reply")
        return self.draft, "Question; P4; answered from F02."

    def write_handoff(self, ticket: Ticket, **kwargs: object) -> tuple[str, str]:
        self.seen.append(ticket.body)
        self._call("write_handoff")
        return "Customer wants a refund for August.", "Refund request; P2; client rule."


class FakeJev:
    def __init__(self, tracker: CostTracker, decision=None, scores=None, fail: bool = False):
        self.tracker = tracker
        self.decision = decision or jev_decision()
        self.scores = scores or GOOD_SCORES
        self.fail = fail
        self.states: list[dict] = []

    def decide(self, state: dict):
        self.states.append(state)
        self.tracker.calls.append(CallRecord("jev", "jev", "jev-1.14-x", 2000, 0, 0.00008, True))
        if self.fail:
            raise ModelError("jev down")
        return self.decision

    def verify_reply(self, state: dict) -> dict[str, float]:
        self.states.append(state)
        return self.scores


def make_agent(**kwargs):
    tracker = CostTracker(limit_usd=1.0)
    deepseek = FakeDeepSeek(tracker, **kwargs.pop("ds", {}))
    jev = FakeJev(tracker, **kwargs.pop("jev", {}))
    return Agent(deepseek, jev, FAQ, tracker, Thresholds()), deepseek, jev


def test_faq_question_gets_a_verified_reply():
    agent, deepseek, jev = make_agent()
    decided = agent.decide(HOW_TO)
    decision = decided.decision
    assert decision.action == "reply"
    assert decision.draft_reply.startswith("Hoi")
    assert decision.language == "nl" and decision.priority == "P4"
    assert decided.trace["rule"] == "faq_reply"
    assert decided.trace["verify"]["grounded"] == 0.95
    assert decided.trace["quotes_dropped"] == 1
    assert decided.trace["model_calls"] == 3
    assert decided.trace["cost_usd"] == pytest.approx(0.00028)
    # Jev read the ticket itself, not only the interpreter's evidence.
    assert jev.states[0]["ticket"]["body"] == HOW_TO.body
    assert "help_centre_faq" in jev.states[0]


def test_trace_contains_no_ticket_or_reply_text():
    agent, _, _ = make_agent()
    trace_text = json.dumps(agent.decide(HOW_TO).trace)
    assert "medewerker" not in trace_text.lower()
    assert "Hoi" not in trace_text


def test_refund_hands_off_without_writing_a_reply():
    agent, deepseek, _ = make_agent(
        jev={"decision": jev_decision(priority="P2", refund_or_disputed_charge=0.9)}
    )
    decision = agent.decide(HOW_TO).decision
    assert decision.action == "handoff"
    assert decision.handoff_note == "Customer wants a refund for August."
    assert "write_reply" not in deepseek.called


def test_empty_ticket_closes_without_model_calls():
    agent, deepseek, _ = make_agent()
    decided = agent.decide(Ticket(id="T012", subject="Re:", body="  "))
    assert decided.decision.action == "close"
    assert decided.trace["model_calls"] == 0
    assert deepseek.called == []


def test_spam_closes():
    agent, _, _ = make_agent(jev={"decision": jev_decision(spam=0.97, answerable=0.0)})
    decision = agent.decide(HOW_TO).decision
    assert decision.action == "close" and decision.confidence == 0.97


def test_jev_failure_hands_off_with_template():
    agent, deepseek, _ = make_agent(jev={"fail": True})
    decided = agent.decide(HOW_TO)
    assert decided.decision.action == "handoff"
    assert decided.decision.language == "de"  # interpreter's guess as fallback
    assert decided.decision.category == "other"
    assert "decision model failed" in decided.decision.handoff_note
    assert decided.trace["rule"] == "decision_model_failed"
    assert "write_handoff" not in deepseek.called


def test_interpreter_failure_still_decides_but_cannot_reply():
    agent, _, jev = make_agent(ds={"fail": {"interpret"}})
    decided = agent.decide(HOW_TO)
    assert decided.decision.action == "handoff"
    assert decided.trace["rule"] == "no_requests"
    assert "unavailable" in jev.states[0]["interpreter_evidence"]


def test_failed_verification_turns_reply_into_handoff():
    agent, _, _ = make_agent(jev={"scores": {**GOOD_SCORES, "grounded": 0.4}})
    decided = agent.decide(HOW_TO)
    assert decided.decision.action == "handoff"
    assert decided.trace["rule"] == "reply_rejected"
    assert "not support" in decided.trace["reply_rejected"]


def test_promise_in_draft_is_rejected():
    agent, _, _ = make_agent(jev={"scores": {**GOOD_SCORES, "commitment": 0.8}})
    assert agent.decide(HOW_TO).decision.action == "handoff"


def test_payment_details_in_draft_are_rejected():
    agent, _, _ = make_agent(ds={"draft": "Uw kaart 4111 1111 1111 1111 is gewijzigd."})
    decided = agent.decide(HOW_TO)
    assert decided.decision.action == "handoff"
    assert "payment details" in decided.trace["reply_rejected"]


def test_models_only_see_redacted_payment_details():
    ticket = HOW_TO.model_copy(update={"body": "Nieuwe kaart: 4111 1111 1111 1111, CVC 123"})
    agent, deepseek, jev = make_agent(
        jev={"decision": jev_decision(priority="P2", cancellation_or_contract_change=0.8)}
    )
    decided = agent.decide(ticket)
    assert decided.trace["payment_details_redacted"] is True
    assert all("4111" not in body and "123" not in body for body in deepseek.seen)
    assert "4111" not in jev.states[0]["ticket"]["body"]


def test_injection_is_noted_and_ignored():
    agent, _, _ = make_agent(jev={"decision": jev_decision(injection_attempt=0.95)})
    decision = agent.decide(HOW_TO).decision
    assert decision.action == "reply"
    assert "instructions aimed at an AI" in decision.reason


def test_handoff_writer_failure_uses_template():
    agent, _, _ = make_agent(
        ds={"fail": {"write_handoff"}},
        jev={"decision": jev_decision(priority="P1", security_incident=0.9)},
    )
    decision = agent.decide(HOW_TO).decision
    assert decision.action == "handoff"
    assert decision.priority == "P1"
    assert "Wants to add an employee." in decision.handoff_note


def test_reply_writer_failure_hands_off():
    agent, _, _ = make_agent(ds={"fail": {"write_reply"}})
    decided = agent.decide(HOW_TO)
    assert decided.decision.action == "handoff"
    assert "writer failed" in decided.trace["reply_rejected"]
