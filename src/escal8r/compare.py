"""Compare two versions of the agent from their trails.

Every number is computed from traces/<run>.jsonl, the same way for both versions:
- a run trail (dry or live) gives behaviour on the practice tickets (no labels);
- eval trails give accuracy and safety on the labelled examples (labels come from
  GET /examples and were stored on each eval root span).
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from escal8r.observability import Trail, check_run
from escal8r.redact import contains_payment_details

FIELDS = ("language", "category", "priority", "action")
STAGES = ("interpret", "jev_decide", "write_reply", "jev_verify", "write_handoff")


def _decide_span(trail: Trail, root: dict[str, Any]) -> dict[str, Any] | None:
    return next((s for s in trail.descendants(root) if s["kind"] == "decide"), None)


def run_metrics(trail: Trail) -> dict[str, Any]:
    """Behaviour, quality signals and operations for one run trail."""
    roots = trail.roots()
    n = len(roots) or 1
    decides = [d for r in roots if (d := _decide_span(trail, r))]
    actions = Counter(d["attrs"].get("action") for d in decides)
    calls = [s for s in trail.spans if s["kind"] == "model_call"]
    replies_tried = [s for s in trail.spans if s["kind"] == "write_reply"]
    verifies = [s for s in trail.spans if s["kind"] == "jev_verify"]
    interprets = [s for s in trail.spans if s["kind"] == "interpret"]
    cost = sum(float(c["attrs"].get("cost_usd") or 0) for c in calls)
    stage_ms = {
        stage: statistics.mean(s["ms"] for s in trail.spans if s["kind"] == stage)
        for stage in STAGES
        if any(s["kind"] == stage for s in trail.spans)
    }
    helpdesk_retries = sum(
        max(0, len(s["attrs"].get("attempts", [])) - 1)
        for s in trail.spans
        if s["kind"] == "helpdesk"
    )
    return {
        "tickets": len(roots),
        "reply": actions.get("reply", 0),
        "handoff": actions.get("handoff", 0),
        "close": actions.get("close", 0),
        "handoff_rate": actions.get("handoff", 0) / n,
        "priority": dict(sorted(Counter(d["attrs"].get("priority") for d in decides).items())),
        "category": dict(sorted(Counter(d["attrs"].get("category") for d in decides).items())),
        "rules": dict(Counter(d["attrs"].get("rule") for d in decides).most_common()),
        "routes": dict(
            Counter(
                s["attrs"]["route"]
                for s in trail.spans
                if s["kind"] == "rules" and s["attrs"].get("route")
            ).most_common()
        ),
        "replies_drafted": len(replies_tried),
        "verify_passed": sum(1 for v in verifies if v["attrs"].get("passed")),
        "verify_rejected": sum(1 for v in verifies if v["attrs"].get("passed") is False),
        "quotes_dropped": sum(int(s["attrs"].get("quotes_dropped") or 0) for s in interprets),
        "mean_confidence": round(
            statistics.mean(float(d["attrs"].get("confidence") or 0) for d in decides), 3
        )
        if decides
        else 0.0,
        "cost_usd": round(cost, 6),
        "cost_per_100_usd": round(100 * cost / n, 4),
        "model_calls_per_ticket": round(len(calls) / n, 2),
        "tokens_in_per_ticket": round(sum(c["attrs"].get("input_tokens") or 0 for c in calls) / n),
        "tokens_out_per_ticket": round(
            sum(c["attrs"].get("output_tokens") or 0 for c in calls) / n
        ),
        "mean_ticket_seconds": round(statistics.mean(r["ms"] for r in roots) / 1000, 2)
        if roots
        else 0.0,
        "mean_stage_ms": {k: round(v) for k, v in stage_ms.items()},
        "model_retries": sum(max(0, len(c["attrs"].get("attempts", [])) - 1) for c in calls),
        "helpdesk_retries": helpdesk_retries,
        "posts": sum(
            1 for s in trail.spans if s["kind"] == "helpdesk" and s["attrs"].get("method") == "POST"
        ),
        "error_spans": sum(1 for s in trail.spans if s["status"] == "error"),
        "trace_check_score": check_run(trail, None)["score"],
    }


def eval_metrics(trails: list[Trail]) -> dict[str, Any]:
    """Accuracy and safety on labelled examples, averaged over repeated eval runs."""
    per_run = []
    for trail in trails:
        roots = trail.roots()
        n = len(roots) or 1
        correct = [r["attrs"].get("correct") or {} for r in roots]
        expected = [r["attrs"].get("expected") or {} for r in roots]
        predicted = [r["attrs"].get("action") for r in roots]
        texts = [
            value
            for r in roots
            if (d := _decide_span(trail, r))
            for value in d.get("text", {}).values()
        ]
        per_run.append(
            {
                **{f"{f}_accuracy": sum(c.get(f, False) for c in correct) / n for f in FIELDS},
                "all_fields_correct": sum(all(c.values()) for c in correct if c),
                "unsafe_replies": sum(
                    p == "reply" and e.get("action") != "reply"
                    for p, e in zip(predicted, expected, strict=True)
                ),
                "wrong_closes": sum(
                    p == "close" and e.get("action") != "close"
                    for p, e in zip(predicted, expected, strict=True)
                ),
                "missed_replies": sum(
                    e.get("action") == "reply" and p != "reply"
                    for p, e in zip(predicted, expected, strict=True)
                ),
                "payment_leaks": sum(contains_payment_details(t) for t in texts),
                "handoff_rate": predicted.count("handoff") / n,
            }
        )
    if not per_run:
        return {}
    return {key: round(statistics.mean(run[key] for run in per_run), 3) for key in per_run[0]} | {
        "runs": len(per_run)
    }


def ticket_diff(a: Trail, b: Trail) -> list[dict[str, Any]]:
    """Tickets whose decision changed between the two runs."""

    def decisions(trail: Trail) -> dict[str, dict[str, Any]]:
        found = {}
        for root in trail.roots():
            decide = _decide_span(trail, root)
            if decide:
                found[root["ticket"]] = decide["attrs"]
        return found

    before, after = decisions(a), decisions(b)
    changes = []
    for ticket in sorted(before.keys() & after.keys()):
        changed = {
            key: (before[ticket].get(key), after[ticket].get(key))
            for key in ("action", "priority", "category", "rule")
            if before[ticket].get(key) != after[ticket].get(key)
        }
        if changed:
            changes.append({"ticket": ticket, **changed})
    return changes


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if value < 1 else f"{value:.2f}"
    if isinstance(value, dict):
        return ", ".join(f"{k} {v}" for k, v in value.items()) or "-"
    return str(value)


def _delta(a: Any, b: Any) -> str:
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        diff = b - a
        if diff == 0:
            return "="
        return f"{'+' if diff > 0 else ''}{_fmt(diff) if isinstance(diff, float) else diff}"
    return "" if a == b else "changed"


def _row(name: str, a: Any, b: Any) -> str:
    return f"| {name} | {_fmt(a)} | {_fmt(b)} | {_delta(a, b)} |"


def format_comparison(
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    eval_a: dict[str, Any],
    eval_b: dict[str, Any],
    diff: list[dict[str, Any]],
) -> str:
    lines = ["| Metric | A | B | Δ |", "|---|---|---|---|"]
    if eval_a and eval_b:
        lines.append(
            f"| **Labelled examples** (mean of {eval_a['runs']} vs {eval_b['runs']} runs) | | | |"
        )
        for key in eval_a:
            if key != "runs":
                lines.append(_row(key, eval_a[key], eval_b[key]))
    lines.append(f"| **Practice tickets** ({run_a['tickets']} vs {run_b['tickets']}) | | | |")
    for key in run_a:
        if key != "tickets":
            lines.append(_row(key, run_a[key], run_b.get(key)))
    lines += ["", f"Tickets whose decision changed: {len(diff)}", ""]
    if diff:
        lines += ["| Ticket | Action | Priority | Category | Rule |", "|---|---|---|---|---|"]
        for change in diff:
            cells = [
                f"{change[k][0]} → {change[k][1]}" if k in change else ""
                for k in ("action", "priority", "category", "rule")
            ]
            lines.append(f"| {change['ticket']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
