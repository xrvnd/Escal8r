"""Plain-code action rules over Jev's answers. No model calls, no ticket text.

Gate order matters: always-human topics are checked before spam, so spam that
mentions a refund is handed off rather than closed.
"""

from __future__ import annotations

from dataclasses import dataclass

from escal8r.config import Thresholds
from escal8r.jev import ALWAYS_HUMAN_GATES, JevDecision
from escal8r.models import Action

URGENT_PRIORITIES = ("P1", "P2")  # policy: urgent tickets always go to a human


@dataclass(frozen=True)
class RuleOutcome:
    action: Action
    rule: str  # short machine-readable name of the rule that fired
    why: str  # one English sentence, used in reasons and handoff notes


def pick_action(jev: JevDecision, t: Thresholds, *, has_requests: bool) -> RuleOutcome:
    """Choose reply / handoff / close. `has_requests`: the interpreter produced at least one
    verified request, which the reply writer needs as input."""
    human = [gate for gate in ALWAYS_HUMAN_GATES if jev.gates[gate] >= t.always_human]
    if human:
        topics = ", ".join(gate.replace("_", " ") for gate in human)
        return RuleOutcome("handoff", "always_human", f"Client rule: {topics} goes to a human.")

    if jev.gates["spam"] >= t.spam:
        return RuleOutcome("close", "spam", "Spam, not a customer message.")

    unsure = [
        name
        for name, answer in (
            ("language", jev.language),
            ("category", jev.category),
            ("priority", jev.priority),
        )
        if answer.confidence < t.choice_confidence
    ]
    if unsure:
        return RuleOutcome(
            "handoff", "low_confidence", f"Unsure about {', '.join(unsure)}; a human should check."
        )

    if jev.priority.choice in URGENT_PRIORITIES:
        return RuleOutcome(
            "handoff", "urgent", f"Priority {jev.priority.choice}: urgent tickets go to a human."
        )

    if jev.gates["answerable"] < t.answerable or jev.gates["covered"] < t.covered:
        return RuleOutcome(
            "handoff", "not_in_faq", "The help centre does not fully answer this ticket."
        )

    if not has_requests:
        return RuleOutcome(
            "handoff", "no_requests", "No customer request could be extracted and verified."
        )

    return RuleOutcome("reply", "faq_reply", "Answered from the help centre.")


def reply_check_failure(scores: dict[str, float], t: Thresholds) -> str | None:
    """Why a draft reply must not be sent, or None if it passes verification."""
    if scores["grounded"] < t.grounded:
        return "draft makes claims the FAQ does not support"
    if scores["reply_language"] < t.reply_language:
        return "draft is not in the customer's language"
    if scores["commitment"] >= t.commitment:
        return "draft promises something the FAQ does not"
    return None
