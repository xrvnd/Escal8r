import json

from escal8r.evaluation import evaluate, format_report
from escal8r.models import Decided, Decision, Example, Ticket
from escal8r.ticketlog import TicketLog


def example(eid: str, action: str, category: str = "question", priority: str = "P4") -> Example:
    return Example.model_validate(
        {
            "id": eid,
            "subject": "s",
            "body": "b",
            "label": {
                "language": "nl",
                "category": category,
                "priority": priority,
                "action": action,
                "reason": "label reason",
            },
        }
    )


def decided(action: str, category: str = "question", priority: str = "P4", text: str = "x"):
    fields = {"reply": {"draft_reply": text}, "handoff": {"handoff_note": text}, "close": {}}
    return Decided(
        decision=Decision(
            action=action,
            language="nl",
            category=category,
            priority=priority,
            reason="r",
            **fields[action],
        ),
        trace={"rule": "t", "cost_usd": 0.0005, "quotes_dropped": 1},
    )


def test_scores_fields_safety_and_cost():
    examples = [
        example("E01", "reply"),
        example("E02", "handoff", "billing", "P2"),
        example("E03", "reply"),
        example("E04", "close", "other"),
    ]
    predictions = {
        "E01": decided("reply"),
        "E02": decided("reply", "billing", "P2"),  # unsafe: should be a human
        "E03": decided("handoff"),  # missed reply
        "E04": decided("close", "other"),
    }
    seen: list[Ticket] = []

    def decide(ticket: Ticket) -> Decided:
        seen.append(ticket)
        return predictions[ticket.id]

    report = evaluate(examples, decide)
    assert report["accuracy"]["action"] == 0.5
    assert report["accuracy"]["category"] == 1.0
    assert report["safety"]["unsafe_replies"] == ["E02"]
    assert report["safety"]["wrong_closes"] == []
    assert report["usefulness"]["missed_replies"] == ["E03"]
    assert report["usefulness"]["handoff_rate"] == 0.25
    assert report["usefulness"]["quotes_dropped"] == 4
    assert report["cost"]["per_100_tickets_usd"] == 0.05
    # The label never reaches the agent.
    assert all(type(t) is Ticket for t in seen)
    assert "E02" in format_report(report)


def test_payment_leak_is_flagged():
    report = evaluate(
        [example("E01", "reply")], lambda t: decided("reply", text="kaart 4111 1111 1111 1111")
    )
    assert report["safety"]["payment_leaks"] == ["E01"]


def test_log_line_has_decision_but_no_text(tmp_path):
    log = TicketLog(tmp_path, run_id="test")
    d = decided("reply", text="Hoi Anouk, ga naar Team.")
    log.write("T005", "posted", d.decision, d.trace)
    log.write("T008", "post_failed", error="card 4111 1111 1111 1111 rejected")

    lines = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["ticket"] == "T005" and lines[0]["action"] == "reply"
    assert "Anouk" not in log.path.read_text()
    assert "4111" not in log.path.read_text()


def test_eval_stops_on_setup_error_instead_of_scoring_fallbacks():
    import pytest

    from escal8r.openrouter import SetupError

    def decide(ticket: Ticket) -> Decided:
        raise SetupError("401 bad key")

    with pytest.raises(SetupError):
        evaluate([example("E01", "reply")], decide)
