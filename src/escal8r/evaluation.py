"""Eval on the helpdesk's 10 labelled examples (GET /examples). Never posts anything.

Labels come only from the helpdesk; the models under test never see them.
Scores: per-field accuracy, safety (unsafe replies, wrong closes, payment leaks),
usefulness (missed replies, handoff rate) and cost.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from escal8r.models import Decided, Example, Ticket
from escal8r.redact import contains_payment_details

FIELDS = ("language", "category", "priority", "action")


@dataclass
class ExampleResult:
    id: str
    label: dict[str, str]
    predicted: dict[str, str]
    correct: dict[str, bool]
    unsafe_reply: bool  # replied where a human or close was expected
    wrong_close: bool  # closed a ticket that should not be closed
    payment_leak: bool
    reason: str
    draft_reply: str | None
    handoff_note: str | None
    trace: dict[str, Any]


def evaluate(examples: list[Example], decide: Callable[[Ticket], Decided]) -> dict[str, Any]:
    results = [_score_one(example, decide) for example in examples]
    total = len(results) or 1
    cost = sum(float(r.trace.get("cost_usd", 0.0)) for r in results)
    predicted_actions = [r.predicted["action"] for r in results]
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "examples": len(results),
        "accuracy": {f: sum(r.correct[f] for r in results) / total for f in FIELDS},
        "all_fields_correct": sum(all(r.correct.values()) for r in results),
        "safety": {
            "unsafe_replies": [r.id for r in results if r.unsafe_reply],
            "wrong_closes": [r.id for r in results if r.wrong_close],
            "payment_leaks": [r.id for r in results if r.payment_leak],
        },
        "usefulness": {
            "missed_replies": [
                r.id
                for r in results
                if r.label["action"] == "reply" and r.predicted["action"] != "reply"
            ],
            "handoff_rate": predicted_actions.count("handoff") / total,
            "label_handoff_rate": [r.label["action"] for r in results].count("handoff") / total,
            "quotes_dropped": sum(int(r.trace.get("quotes_dropped", 0)) for r in results),
        },
        "cost": {
            "total_usd": round(cost, 6),
            "per_ticket_usd": round(cost / total, 6),
            "per_100_tickets_usd": round(100 * cost / total, 4),
            "estimated": any(r.trace.get("cost_estimated") for r in results),
        },
        "results": [asdict(r) for r in results],
    }


def _score_one(example: Example, decide: Callable[[Ticket], Decided]) -> ExampleResult:
    # The agent gets a plain Ticket: the label never reaches the models.
    ticket = Ticket.model_validate(example.model_dump(exclude={"label"}))
    decided = decide(ticket)
    decision = decided.decision
    label = example.label
    predicted = {f: str(getattr(decision, f)) for f in FIELDS}
    expected = {f: str(getattr(label, f)) for f in FIELDS}
    texts = [decision.draft_reply or "", decision.handoff_note or "", decision.reason]
    return ExampleResult(
        id=example.id,
        label={**expected, "reason": label.reason},
        predicted=predicted,
        correct={f: predicted[f] == expected[f] for f in FIELDS},
        unsafe_reply=decision.action == "reply" and label.action != "reply",
        wrong_close=decision.action == "close" and label.action != "close",
        payment_leak=any(contains_payment_details(text) for text in texts),
        reason=decision.reason,
        draft_reply=decision.draft_reply,
        handoff_note=decision.handoff_note,
        trace=decided.trace,
    )


def save_report(report: dict[str, Any], directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def format_report(report: dict[str, Any]) -> str:
    accuracy = report["accuracy"]
    safety = report["safety"]
    usefulness = report["usefulness"]
    cost = report["cost"]
    lines = [
        f"examples: {report['examples']}   all fields correct: {report['all_fields_correct']}",
        "accuracy: " + "  ".join(f"{f} {accuracy[f]:.0%}" for f in FIELDS),
        f"safety:   unsafe replies {safety['unsafe_replies'] or 'none'}   "
        f"wrong closes {safety['wrong_closes'] or 'none'}   "
        f"payment leaks {safety['payment_leaks'] or 'none'}",
        f"useful:   missed replies {usefulness['missed_replies'] or 'none'}   "
        f"handoff rate {usefulness['handoff_rate']:.0%} "
        f"(labels {usefulness['label_handoff_rate']:.0%})   "
        f"quotes dropped {usefulness['quotes_dropped']}",
        f"cost:     ${cost['total_usd']:.4f} total, "
        f"${cost['per_100_tickets_usd']:.4f} per 100 tickets"
        + (" (partly estimated)" if cost["estimated"] else ""),
        "",
        f"{'id':<5}{'language':<12}{'category':<28}{'priority':<10}{'action':<20}rule",
    ]
    for r in report["results"]:
        cells = [
            r["predicted"][f] if r["correct"][f] else f"{r['predicted'][f]}≠{r['label'][f]}"
            for f in FIELDS
        ]
        lines.append(
            f"{r['id']:<5}{cells[0]:<12}{cells[1]:<28}{cells[2]:<10}{cells[3]:<20}"
            f"{r['trace'].get('rule', '')}"
        )
    return "\n".join(lines)
