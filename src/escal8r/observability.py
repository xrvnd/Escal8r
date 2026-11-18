"""Read trails back: render one ticket's timeline, and check a run's trails are complete.

`render_trail` answers "what happened to ticket X?": every stage, model call and
helpdesk call, with timings, retries, models, tokens, cost and (optionally) the
redacted prompts and outputs.

`check_run` is the observability eval: for every ticket in a run it checks that the
trail is there, complete for the path the ticket took, consistent with the committed
log, and free of leaks. A failed check means a decision we could not fully explain.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from escal8r.redact import contains_payment_details
from escal8r.tracing import load_spans

Span = dict[str, Any]

# Stages every decided ticket must show, by the rule that decided it.
_BASE = frozenset({"redact", "interpret", "jev_decide", "rules"})
REQUIRED_STAGES: dict[str, frozenset[str]] = {
    "empty": frozenset({"redact"}),
    "decision_model_failed": frozenset({"redact", "interpret", "jev_decide", "templated_handoff"}),
    "spam": _BASE,
    "faq_reply": _BASE | {"write_reply", "jev_verify"},
}
HANDOFF_STAGES = _BASE | {"write_handoff"}  # every other rule ends in a written handoff
STAGES_WITH_MODEL_CALL = {"interpret", "jev_decide", "write_reply", "jev_verify", "write_handoff"}
POSTED_OUTCOMES = {"posted", "already_handled", "outcome_unknown", "post_failed"}
OUTCOMES_NEEDING_CAUSE = {"decision_failed", "run_stopped", "outcome_unknown", "post_failed"}
# Tickets that legitimately have no complete decision trail.
NO_DECISION_OUTCOMES = {"recovered", "skipped_done_locally", "run_stopped", "decision_failed"}
MODEL_CALL_FIELDS = ("model_served", "input_tokens", "cost_usd")
LOG_TEXT_KEYS = {"draft_reply", "handoff_note", "reason", "text", "subject", "body"}
CLOCK_SLACK = timedelta(milliseconds=50)


# -- loading -----------------------------------------------------------------


def latest_trace(directory: Path, ticket_id: str | None = None) -> Path | None:
    """Newest trace file, optionally the newest one that contains `ticket_id`."""
    for path in sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        if ticket_id is None or f'"ticket": "{ticket_id}"' in path.read_text(encoding="utf-8"):
            return path
    return None


@dataclass
class Trail:
    spans: list[Span]
    children: dict[str | None, list[Span]] = field(default_factory=lambda: defaultdict(list))
    by_id: dict[str, Span] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for span in self.spans:
            self.by_id[span["span"]] = span
            self.children[span["parent"]].append(span)
        for kids in self.children.values():
            kids.sort(key=lambda s: s["start"])

    def roots(self, ticket_id: str | None = None) -> list[Span]:
        return [
            s
            for s in self.spans
            if s["kind"] == "ticket" and (ticket_id is None or s["ticket"] == ticket_id)
        ]

    def descendants(self, span: Span) -> list[Span]:
        found, stack = [], list(self.children.get(span["span"], []))
        while stack:
            child = stack.pop()
            found.append(child)
            stack.extend(self.children.get(child["span"], []))
        return found

    def cost(self, span: Span) -> float:
        return sum(
            float(s["attrs"].get("cost_usd") or 0)
            for s in self.descendants(span)
            if s["kind"] == "model_call"
        )


def load_trail(path: Path) -> Trail:
    return Trail(load_spans(path))


# -- rendering -----------------------------------------------------------------


def _fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


def _describe(span: Span) -> str:
    attrs = span["attrs"]
    kind = span["kind"]
    if kind == "model_call":
        parts = [
            attrs.get("purpose", ""),
            f"{attrs.get('model_served') or attrs.get('model_requested')}",
            f"{attrs.get('input_tokens', '?')}→{attrs.get('output_tokens', '?')} tok",
            f"${float(attrs.get('cost_usd') or 0):.6f}"
            + ("" if attrs.get("cost_reported", True) else " (est.)"),
            f"attempts {attrs.get('attempts', [])}",
        ]
        if attrs.get("generation_id"):
            parts.append(str(attrs["generation_id"]))
        if attrs.get("answers"):
            parts.append(f"answers {json.dumps(attrs['answers'])}")
        return " | ".join(parts)
    if kind == "helpdesk":
        attempts = ", ".join(
            str(a.get("result", "?"))
            + (f" (retry {a['retry_in_s']}s)" if "retry_in_s" in a else "")
            for a in attrs.get("attempts", [])
        )
        wait = attrs.get("limiter_wait_s")
        return f"{attrs.get('method')} {attrs.get('path')} | [{attempts}]" + (
            f" | rate-limit wait {wait}s" if wait else ""
        )
    shown = {k: v for k, v in attrs.items() if k not in {"gates", "choices"}}
    text = " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in shown.items())
    if "gates" in attrs:
        raised = {k: v for k, v in attrs["gates"].items() if v >= 0.3}
        choices = {k: f"{v['choice']}@{v['confidence']}" for k, v in attrs["choices"].items()}
        text = f"choices={choices} gates≥0.3={raised}"
        if "route" in attrs:
            route = attrs["route"]
            text += (
                f" route={route['top']} human_mass={route['human_mass']} "
                f"margin={route['margin']} users_affected={attrs.get('users_affected')}"
            )
    return text


def render_trail(trail: Trail, ticket_id: str, *, show_text: bool = False, width: int = 600) -> str:
    roots = trail.roots(ticket_id)
    if not roots:
        return f"no trail for {ticket_id}"
    lines: list[str] = []
    for root in roots:
        attrs = root["attrs"]
        lines.append(
            f"{ticket_id}  run {root['run']}  outcome={attrs.get('outcome')} "
            f"action={attrs.get('action')} rule={attrs.get('rule')}  "
            f"{_fmt_ms(root['ms'])}  ${trail.cost(root):.6f}"
            + (f"  ERROR {root['error']}" if root.get("error") else "")
        )
        _render_children(trail, root, "", lines, show_text, width)
    return "\n".join(lines)


def _render_children(
    trail: Trail, span: Span, prefix: str, lines: list[str], show_text: bool, width: int
) -> None:
    kids = trail.children.get(span["span"], [])
    for index, child in enumerate(kids):
        last = index == len(kids) - 1
        branch, cont = ("└─ ", "   ") if last else ("├─ ", "│  ")
        status = f"  ✗ {child['error']}" if child.get("error") else ""
        lines.append(
            f"{prefix}{branch}{child['kind']} {_fmt_ms(child['ms'])}  {_describe(child)}{status}"
        )
        if show_text:
            for name, value in child.get("text", {}).items():
                snippet = value if len(value) <= width else value[:width] + " …"
                for i, text_line in enumerate(snippet.splitlines() or [""]):
                    label = f"[{name}] " if i == 0 else " " * (len(name) + 3)
                    lines.append(f"{prefix}{cont}    {label}{text_line}")
        _render_children(trail, child, prefix + cont, lines, show_text, width)


# -- completeness eval -----------------------------------------------------------


@dataclass
class Check:
    name: str
    description: str
    passed: int = 0
    failures: list[str] = field(default_factory=list)

    def record(self, ok: bool, failure: str) -> None:
        if ok:
            self.passed += 1
        else:
            self.failures.append(failure)


def _end(span: Span) -> datetime:
    return datetime.fromisoformat(span["start"]) + timedelta(milliseconds=span["ms"])


def check_run(trail: Trail, log_lines: list[dict[str, Any]] | None) -> dict[str, Any]:
    checks = {
        name: Check(name, description)
        for name, description in (
            ("log_matches_trail", "one log line per ticket, pointing at its trail root"),
            ("stages_complete", "every stage the ticket's path requires is in its trail"),
            ("model_calls_complete", "every model call records model, tokens and cost"),
            ("posts_traced", "every posted decision has its helpdesk call and attempts"),
            ("cost_reconciles", "trail model costs equal the cost in the log line"),
            ("failures_explained", "every failure or non-success outcome carries a cause"),
            ("tree_intact", "no orphan spans; children share ticket and fit in parent's time"),
            ("no_leaks", "no text fields in the log; no payment details in log or trail"),
        )
    }
    roots = trail.roots()
    logs_by_ticket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in log_lines or []:
        logs_by_ticket[line["ticket"]].append(line)

    for root in roots:
        ticket = root["ticket"]
        tree = trail.descendants(root)
        kinds = {s["kind"] for s in tree}
        attrs = root["attrs"]
        outcome = attrs.get("outcome")

        if log_lines is not None:
            lines = logs_by_ticket.get(ticket, [])
            checks["log_matches_trail"].record(
                len(lines) == 1 and lines[0].get("span") == root["span"],
                f"{ticket}: {len(lines)} log lines for trail root {root['span']}",
            )
            if lines and "trace" in lines[0]:
                logged = float(lines[0]["trace"].get("cost_usd") or 0)
                traced = round(trail.cost(root), 6)
                checks["cost_reconciles"].record(
                    abs(logged - traced) < 1e-6, f"{ticket}: log ${logged} vs trail ${traced}"
                )

        decide = next((s for s in tree if s["kind"] == "decide"), None)
        if outcome in NO_DECISION_OUTCOMES or attrs.get("resent_from_state"):
            pass  # no fresh decision was made, so there are no stages to require
        elif decide is not None:
            rule = decide["attrs"].get("rule", "")
            required = REQUIRED_STAGES.get(rule, HANDOFF_STAGES)
            missing = sorted(required - kinds)
            checks["stages_complete"].record(not missing, f"{ticket} ({rule}): missing {missing}")
            for stage in (s for s in tree if s["kind"] in STAGES_WITH_MODEL_CALL):
                has_call = any(c["kind"] == "model_call" for c in trail.descendants(stage))
                checks["model_calls_complete"].record(
                    has_call, f"{ticket}: stage {stage['kind']} has no model_call span"
                )
        else:
            checks["stages_complete"].record(False, f"{ticket}: no decide span ({outcome})")

        for call in (s for s in tree if s["kind"] == "model_call" and s["status"] == "ok"):
            missing_fields = [f for f in MODEL_CALL_FIELDS if call["attrs"].get(f) is None]
            checks["model_calls_complete"].record(
                not missing_fields,
                f"{ticket}: {call['attrs'].get('purpose')} call missing {missing_fields}",
            )

        if outcome in POSTED_OUTCOMES:
            posts = [
                s for s in tree if s["kind"] == "helpdesk" and s["attrs"].get("method") == "POST"
            ]
            checks["posts_traced"].record(
                len(posts) == 1 and bool(posts[0]["attrs"].get("attempts")),
                f"{ticket}: {len(posts)} POST spans for outcome {outcome}",
            )

        for span in [root, *tree]:
            if span["status"] == "error":
                checks["failures_explained"].record(
                    bool(span.get("error")), f"{ticket}: {span['kind']} failed without a cause"
                )
        if outcome in OUTCOMES_NEEDING_CAUSE:
            explained = any(s.get("error") for s in [root, *tree])
            checks["failures_explained"].record(explained, f"{ticket}: '{outcome}' has no cause")

        root_start, root_end = datetime.fromisoformat(root["start"]), _end(root)
        for span in tree:
            parent = trail.by_id.get(span["parent"])
            ok = (
                parent is not None
                and span["ticket"] == ticket
                and span["run"] == root["run"]
                and datetime.fromisoformat(span["start"]) >= root_start - CLOCK_SLACK
                and _end(span) <= root_end + CLOCK_SLACK
            )
            checks["tree_intact"].record(ok, f"{ticket}: span {span['span']} ({span['kind']})")

    # Every span, including run-level ones (e.g. fetching tickets) outside any ticket trail.
    for span in trail.spans:
        leaked = [k for k, v in span.get("text", {}).items() if contains_payment_details(v)]
        checks["no_leaks"].record(
            not leaked, f"{span['ticket'] or 'run'}: payment details in {span['kind']} {leaked}"
        )

    orphans = [s for s in trail.spans if s["parent"] and s["parent"] not in trail.by_id]
    checks["tree_intact"].record(not orphans, f"{len(orphans)} orphan spans")

    if log_lines is not None:
        traced_tickets = {r["ticket"] for r in roots}
        for ticket, lines in logs_by_ticket.items():
            checks["log_matches_trail"].record(
                ticket in traced_tickets, f"{ticket}: log line without a trail"
            )
            for line in lines:
                bad_keys = sorted(LOG_TEXT_KEYS & set(line))
                leak = contains_payment_details(json.dumps(line, ensure_ascii=False))
                checks["no_leaks"].record(
                    not bad_keys and not leak,
                    f"{ticket}: log line has {bad_keys or 'payment details'}",
                )

    results = {
        name: {
            "description": c.description,
            "passed": c.passed,
            "failed": len(c.failures),
            "failures": c.failures,
        }
        for name, c in checks.items()
    }
    total = sum(c.passed + len(c.failures) for c in checks.values())
    failed = sum(len(c.failures) for c in checks.values())
    return {
        "run": roots[0]["run"] if roots else None,
        "tickets": len(roots),
        "assertions": total,
        "failed": failed,
        "score": round((total - failed) / total, 4) if total else 0.0,
        "checks": results,
    }


def format_check(report: dict[str, Any]) -> str:
    lines = [
        f"run {report['run']}: {report['tickets']} tickets, {report['assertions']} assertions, "
        f"{report['failed']} failed, score {report['score']:.1%}",
    ]
    for name, result in report["checks"].items():
        mark = "PASS" if not result["failed"] else "FAIL"
        lines.append(
            f"[{mark}] {name:<22} {result['passed']:>4} ok {result['failed']:>3} failed  "
            f"{result['description']}"
        )
        lines.extend(f"         - {failure}" for failure in result["failures"][:10])
    return "\n".join(lines)
