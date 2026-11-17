"""Shared builders for Jev answers in the documented Decisions response shape."""

from __future__ import annotations

from typing import Any

from escal8r.jev import DECISION_QUESTIONS, ChoiceAnswer, JevDecision


def jev_answers(
    *,
    language: str = "nl",
    category: str = "question",
    priority: str = "P4",
    confidence: float = 0.9,
    **gates: float,
) -> dict[str, Any]:
    """A raw `answers` map like the Decisions endpoint returns. Gates default to a safe
    'answerable FAQ question': every always-human gate and spam low, answerable/covered high."""
    nouls = {name: 0.02 for name, q in DECISION_QUESTIONS.items() if q["type"] == "noul"} | {
        "answerable": 0.95,
        "covered": 0.95,
    }
    nouls.update(gates)
    answers: dict[str, Any] = {
        name: {"type": "noul", "noul": value} for name, value in nouls.items()
    }
    for name, choice in (("language", language), ("category", category), ("priority", priority)):
        answers[name] = {
            "type": "choice",
            "choice": choice,
            "probabilities": {choice: confidence},
            "confidence": confidence,
        }
    return answers


def jev_decision(confidence: float = 0.9, priority: str = "P4", **gates: float) -> JevDecision:
    raw = jev_answers(confidence=confidence, priority=priority, **gates)
    choice = {
        name: ChoiceAnswer(raw[name]["choice"], confidence, {})
        for name in ("language", "category", "priority")
    }
    nouls = {name: a["noul"] for name, a in raw.items() if a["type"] == "noul"}
    return JevDecision(choice["language"], choice["category"], choice["priority"], nouls)
