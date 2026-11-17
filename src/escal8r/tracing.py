"""Per-ticket trails: nested spans written to traces/<run>.jsonl (gitignored).

A span is one step: a ticket, an agent stage, a model call or a helpdesk call. Only the
code that owns a run (the runner, the eval) picks a Tracer and opens root spans. Everything
below uses the module-level `span()`, which nests under the current span and writes to
*its* tracer, so a call can never land in another run's file. Outside any root span,
`span()` records nothing.

Two levels of detail, on purpose:
- logs/ (committed): one text-free line per ticket, pointing to its root span;
- traces/ (local only): timings, retries, model ids, tokens, cost and the *redacted*
  prompts and outputs, so a single decision can be replayed. All text passes through
  `redact` before it is written, so payment details never reach disk.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from escal8r.redact import redact


@dataclass
class Span:
    run_id: str
    span_id: str
    parent_id: str | None
    ticket_id: str | None
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    text: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    tracer: Tracer | None = field(default=None, repr=False)

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def fail(self, error: str) -> None:
        """Mark the step as failed even though no exception escapes it."""
        self.error = redact(error)


_current: ContextVar[Span | None] = ContextVar("escal8r_span", default=None)


def current_span() -> Span | None:
    return _current.get()


class Tracer:
    def __init__(self, directory: Path | None, run_id: str, *, capture_text: bool = True):
        self.run_id = run_id
        self.capture_text = capture_text
        self.path = directory / f"{run_id}.jsonl" if directory else None
        if directory:
            directory.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(self, kind: str, *, ticket_id: str | None = None, **attrs: Any) -> Iterator[Span]:
        parent = _current.get()
        span = Span(
            run_id=self.run_id,
            span_id=uuid.uuid4().hex[:12],
            parent_id=parent.span_id if parent else None,
            ticket_id=ticket_id or (parent.ticket_id if parent else None),
            kind=kind,
            attrs=dict(attrs),
            tracer=self,
        )
        token = _current.set(span)
        started = datetime.now(UTC)
        clock = time.perf_counter()
        try:
            yield span
        except BaseException as exc:
            span.fail(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _current.reset(token)
            self._write(span, started, time.perf_counter() - clock)

    def text(self, span: Span, name: str, value: str | None) -> None:
        """Attach redacted text to a span (only when text capture is on)."""
        if self.capture_text and value:
            span.text[name] = redact(value)

    def _write(self, span: Span, started: datetime, seconds: float) -> None:
        if self.path is None:
            return
        record = {
            "run": span.run_id,
            "span": span.span_id,
            "parent": span.parent_id,
            "ticket": span.ticket_id,
            "kind": span.kind,
            "start": started.isoformat(timespec="milliseconds"),
            "ms": round(seconds * 1000, 1),
            "status": "error" if span.error else "ok",
            "attrs": span.attrs,
        }
        if span.error:
            record["error"] = span.error
        if span.text:
            record["text"] = span.text
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


NULL_TRACER = Tracer(None, "untraced", capture_text=False)


@contextmanager
def span(kind: str, **attrs: Any) -> Iterator[Span]:
    """A child of the current span, written by the current span's tracer."""
    parent = _current.get()
    tracer = parent.tracer if parent is not None and parent.tracer is not None else NULL_TRACER
    with tracer.span(kind, **attrs) as child:
        yield child


def record_text(target: Span, name: str, value: str | None) -> None:
    """Attach redacted text to `target` if its tracer captures text."""
    if target.tracer is not None:
        target.tracer.text(target, name, value)


def load_spans(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
