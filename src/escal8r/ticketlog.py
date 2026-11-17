"""One JSON line per ticket per run in logs/run-<timestamp>.jsonl.

Lines hold ids, decisions, scores, models and cost. They never hold the ticket text,
the draft reply or the handoff note, so no customer or payment details reach the log.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from escal8r.models import Decision
from escal8r.redact import redact


class TicketLog:
    def __init__(self, directory: Path, run_id: str | None = None):
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"run-{self.run_id}.jsonl"

    def write(
        self,
        ticket_id: str,
        outcome: str,
        decision: Decision | None = None,
        trace: dict[str, Any] | None = None,
        error: str | None = None,
        *,
        span_id: str | None = None,
    ) -> None:
        line: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "run": self.run_id,
            "ticket": ticket_id,
            "outcome": outcome,
        }
        if span_id:
            line["span"] = span_id  # root of this ticket's trail in traces/<run>.jsonl
        if decision is not None:
            line.update(
                action=decision.action,
                language=decision.language,
                category=decision.category,
                priority=decision.priority,
                confidence=decision.confidence,
            )
        if trace:
            line["trace"] = trace
        if error:
            line["error"] = redact(error)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
