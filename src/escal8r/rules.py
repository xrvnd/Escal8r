"""Plain-code rules over Jev's answers. No model calls, no ticket text.

- compute_priority: the brief's P1-P4 table, applied to facts Jev answered.
- pick_action: reply / handoff / close, from the route distribution and the gates.
  Always-human gates are checked first and override everything, so spam that
  mentions a refund is handed off rather than closed.
"""

from __future__ import annotations

from dataclasses import dataclass

from escal8r.config import Thresholds
from escal8r.jev import (
    ALWAYS_HUMAN_GATES,
    FAQ_ROUTES,
    GATE_QUEUES,
    HUMAN_ROUTES,
    ROUTES,
    JevDecision,
)
from escal8r.models import Action, Priority

URGENT_PRIORITIES = ("P1", "P2")  # policy: urgent tickets always go to a human
DEFAULT_QUEUE = "Support"


@dataclass(frozen=True)
class RuleOutcome:
    action: Action
    rule: str  # short machine-readable name of the rule that fired
    why: str  # one English sentence, used in reasons and handoff notes
    route: str  # the route the ticket follows
    queue: str = ""  # team a handoff goes to


def compute_priority(jev: JevDecision, t: Thresholds) -> tuple[Priority, str]:
    """The brief's priority table over Jev's facts. Returns (priority, which row fired)."""

    def yes(fact: str) -> bool:
        return jev.gates[fact] >= t.fact

    many_users = jev.users_affected.score >= t.many_users
    bug = jev.category_code == "bug"

    if yes("security_incident") or yes("legal_threat"):
        return "P1", "security or legal risk"
    if many_users and yes("blocked_now"):
        return "P1", "many users blocked right now"
    if yes("refund_or_disputed_charge"):
        return "P2", "money is disputed"
    if yes("cancellation_or_contract_change") or yes("churn_threat"):
        return "P2", "the contract is at stake"
    if yes("personal_data_request"):
        return "P2", "formal personal-data request"
    if yes("blocked_now") and not yes("workaround_exists"):
        return "P2", "one customer blocked, no workaround"
    if bug and yes("deadline_soon"):
        return "P2", "something is wrong and a deadline is near"
    if yes("cosmetic_only"):
        return "P4", "cosmetic only"
    if bug or yes("blocked_now"):
        return "P3", "something is wrong but there is a workaround"
    return "P4", "no urgency"


def _human(rule: str, why: str, route: str, queue: str = "") -> RuleOutcome:
    return RuleOutcome("handoff", rule, why, route, queue or ROUTES[route].queue or DEFAULT_QUEUE)


def pick_action(
    jev: JevDecision, t: Thresholds, *, has_requests: bool, priority: Priority
) -> RuleOutcome:
    """Choose reply / handoff / close. `has_requests`: the interpreter produced at least one
    verified request, which the reply writer needs as input."""
    best_human, human_p = jev.best_route(HUMAN_ROUTES)
    best_faq, _ = jev.best_route(FAQ_ROUTES)

    gates = [gate for gate in ALWAYS_HUMAN_GATES if jev.gates[gate] >= t.always_human]
    if gates:
        topics = ", ".join(gate.replace("_", " ") for gate in gates)
        return _human(
            "always_human",
            f"Client rule: {topics} goes to a human.",
            best_human,
            GATE_QUEUES[gates[0]],
        )

    spam_route = jev.route.probabilities.get("spam", 0.0)
    if jev.gates["spam"] >= t.spam and spam_route >= t.spam_route:
        return RuleOutcome("close", "spam", "Spam, not a customer message.", "spam")
    if jev.gates["spam"] >= t.spam or spam_route >= t.spam_route:
        return _human("spam_unsure", "Spam signals disagree; a human should check.", "human_other")

    unsure = [
        name
        for name, answer in (("language", jev.language), ("category", jev.category))
        if answer.confidence < t.choice_confidence
    ]
    if unsure:
        return _human(
            "low_confidence", f"Unsure about {', '.join(unsure)}; a human should check.", best_human
        )

    if priority in URGENT_PRIORITIES:
        return _human("urgent", f"Priority {priority}: urgent tickets go to a human.", best_human)

    if jev.route_mass(HUMAN_ROUTES) >= t.human_route:
        return _human("human_route", f"Routed to {best_human.removeprefix('human_')}.", best_human)

    if jev.route_margin() < t.route_margin:
        return _human(
            "ambiguous_route",
            f"Torn between {best_faq} and {best_human} ({human_p:.2f}); a human should check.",
            best_human,
        )

    if jev.gates["answerable"] < t.answerable or jev.gates["covered"] < t.covered:
        return _human(
            "not_in_faq", "The help centre does not fully answer this ticket.", best_human
        )

    if not has_requests:
        return _human(
            "no_requests", "No customer request could be extracted and verified.", best_human
        )

    return RuleOutcome("reply", "faq_reply", "Answered from the help centre.", best_faq)


def reply_sections(route: str, cited: list[str]) -> list[str] | None:
    """FAQ sections the reply may use, or None if route and evidence disagree.

    Jev's route decides which sections are relevant; the interpreter's citations are
    evidence. Overlap narrows the sections; no citations fall back to the route's
    sections; citations that miss the route entirely mean the two disagree.
    """
    allowed = ROUTES[route].faq_ids
    if not cited:
        return list(allowed)
    overlap = [section for section in cited if section in allowed]
    return overlap or None


def reply_check_failure(scores: dict[str, float], t: Thresholds) -> str | None:
    """Why a draft reply must not be sent, or None if it passes verification."""
    if scores["grounded"] < t.grounded:
        return "draft makes claims the FAQ does not support"
    if scores["reply_language"] < t.reply_language:
        return "draft is not in the customer's language"
    if scores["commitment"] >= t.commitment:
        return "draft promises something the FAQ does not"
    return None
