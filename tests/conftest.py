"""Shared builders for Jev answers in the documented Decisions response shape."""

from __future__ import annotations

from typing import Any

from escal8r.jev import (
    DECISION_QUESTIONS,
    USERS_AFFECTED_SCALE,
    JevDecision,
    parse_choice,
    parse_noul,
    parse_score,
)


def jev_answers(
    *,
    language: str = "nl",
    category: str = "question",
    route: str = "faq_team_and_rota",
    route_probabilities: dict[str, float] | None = None,
    users_affected: float = 0.0,
    confidence: float = 0.9,
    **gates: float,
) -> dict[str, Any]:
    """A raw `answers` map like the Decisions endpoint returns. Defaults describe a safe,
    answerable FAQ question: every risk gate and fact low, answerable/covered high, and the
    route clearly on one FAQ bucket."""
    nouls = {name: 0.02 for name, q in DECISION_QUESTIONS.items() if q["type"] == "noul"} | {
        "answerable": 0.95,
        "covered": 0.95,
    }
    nouls.update(gates)
    answers: dict[str, Any] = {
        name: {"type": "noul", "noul": value} for name, value in nouls.items()
    }
    for name, choice in (("language", language), ("category", category)):
        answers[name] = {
            "type": "choice",
            "choice": choice,
            "probabilities": {choice: confidence},
            "confidence": confidence,
        }
    probabilities = route_probabilities or {route: 0.9}
    answers["route"] = {
        "type": "choice",
        "choice": max(probabilities, key=lambda name: probabilities[name]),
        "probabilities": probabilities,
        "confidence": confidence,
    }
    answers["users_affected"] = {"type": "score", "score": users_affected, "confidence": 0.9}
    return answers


def jev_decision(**kwargs: Any) -> JevDecision:
    """A parsed JevDecision built through the real parsers."""
    raw = jev_answers(**kwargs)
    criteria = {name: DECISION_QUESTIONS[name]["criteria"] for name in ("language", "category")}
    return JevDecision(
        language=parse_choice(raw, "language", criteria["language"]),
        category=parse_choice(raw, "category", criteria["category"]),
        route=parse_choice(raw, "route", DECISION_QUESTIONS["route"]["criteria"]),
        users_affected=parse_score(raw, "users_affected", USERS_AFFECTED_SCALE),
        gates={
            name: parse_noul(raw, name)
            for name, q in DECISION_QUESTIONS.items()
            if q["type"] == "noul"
        },
    )
