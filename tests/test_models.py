import pytest
from pydantic import ValidationError

from escal8r.models import Decision, Ticket

BASE = {"language": "nl", "category": "question", "priority": "P4", "reason": "r"}


def test_reply_requires_draft():
    with pytest.raises(ValidationError):
        Decision(action="reply", **BASE)


def test_handoff_rejects_draft_reply():
    with pytest.raises(ValidationError):
        Decision(action="handoff", draft_reply="x", handoff_note="y", **BASE)


def test_close_rejects_text():
    with pytest.raises(ValidationError):
        Decision(action="close", handoff_note="y", **BASE)


def test_payload_drops_action_and_empty_fields():
    payload = Decision(action="handoff", handoff_note="Customer wants a refund.", **BASE).payload()
    assert payload == {**BASE, "handoff_note": "Customer wants a refund."}


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        Decision(action="close", confidence=1.5, **BASE)


def test_ticket_null_body_becomes_empty():
    assert Ticket(id="T012", subject="Re:", body=None).body == ""
