"""Processes open tickets: decide -> save in_progress -> post -> save done.

Duplicate-answer protection lives here, not in the models:
- tickets marked done locally are never posted again;
- a ticket left in_progress is re-sent with its stored decision only if the helpdesk
  still lists it as open; otherwise it is marked done;
- a 409 marks the ticket done; a lost POST response leaves it in_progress for the next run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from escal8r.helpdesk import AlreadyHandled, HelpdeskError, OutcomeUnknown, RetriesExhausted
from escal8r.models import Decision, Ticket
from escal8r.state import StateStore, Status

log = logging.getLogger(__name__)

DecideFn = Callable[[Ticket], Decision]


class Helpdesk(Protocol):
    def iter_tickets(self, status: str = "open") -> Iterator[Ticket]: ...
    def post_decision(self, ticket_id: str, decision: Decision) -> dict[str, Any]: ...
    def reset_practice(self) -> dict[str, Any]: ...


@dataclass
class RunSummary:
    posted: list[str] = field(default_factory=list)
    already_handled: list[str] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)
    skipped_done: list[str] = field(default_factory=list)
    outcome_unknown: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    dry_run: dict[str, Decision] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {name: len(value) for name, value in vars(self).items()}


def run(
    client: Helpdesk,
    store: StateStore,
    decide: DecideFn,
    *,
    dry_run: bool = False,
) -> RunSummary:
    """Handle every open ticket once. In dry-run mode nothing is posted or saved."""
    if dry_run:
        return _run(client, store, decide, dry_run=True)
    with store.exclusive():
        return _run(client, store, decide, dry_run=False)


def _run(client: Helpdesk, store: StateStore, decide: DecideFn, *, dry_run: bool) -> RunSummary:
    summary = RunSummary()
    open_tickets = list(client.iter_tickets("open"))
    log.info("%d open tickets", len(open_tickets))

    if not dry_run:
        summary.recovered = store.recover({t.id for t in open_tickets})
        for ticket_id in summary.recovered:
            log.info("%s: recovered, helpdesk already has the decision", ticket_id)

    for ticket in open_tickets:
        entry = store.get(ticket.id)
        if entry and entry.status == Status.DONE:
            # Open in the helpdesk but done locally: the helpdesk was reset without
            # clearing local state. Never risk a second answer; `escal8r reset` clears both.
            log.warning("%s: done locally but open in helpdesk; skipped", ticket.id)
            summary.skipped_done.append(ticket.id)
            continue

        try:
            if entry and entry.status == Status.IN_PROGRESS and entry.decision:
                decision = entry.decision  # re-send exactly what was decided before
            else:
                decision = decide(ticket)
        except Exception:
            log.exception("%s: decision failed", ticket.id)
            if not dry_run:
                store.mark_failed(ticket.id, note="decision failed")
            summary.failed.append(ticket.id)
            continue

        if dry_run:
            summary.dry_run[ticket.id] = decision
            continue

        _post(client, store, ticket.id, decision, summary)

    log.info("run finished: %s", summary.counts())
    return summary


def _post(
    client: Helpdesk,
    store: StateStore,
    ticket_id: str,
    decision: Decision,
    summary: RunSummary,
) -> None:
    store.mark_in_progress(ticket_id, decision)
    try:
        client.post_decision(ticket_id, decision)
    except AlreadyHandled:
        store.mark_done(ticket_id, note="409 already handled")
        summary.already_handled.append(ticket_id)
        log.warning("%s: 409 already handled; marked done", ticket_id)
    except OutcomeUnknown as exc:
        # Stay in_progress: the next run checks whether the helpdesk saved it.
        summary.outcome_unknown.append(ticket_id)
        log.warning("%s: outcome unknown (%s); will verify next run", ticket_id, exc)
    except (HelpdeskError, RetriesExhausted) as exc:
        # The helpdesk saved nothing; safe to try again on the next run.
        store.mark_failed(ticket_id, note=str(exc))
        summary.failed.append(ticket_id)
        log.error("%s: post failed: %s", ticket_id, exc)
    else:
        store.mark_done(ticket_id, note=f"posted {decision.action}")
        summary.posted.append(ticket_id)
        log.info("%s: posted %s", ticket_id, decision.action)


def reset_all(client: Helpdesk, store: StateStore) -> None:
    """Re-open all practice tickets, then clear local state.

    API first: if the reset fails, local state still matches the helpdesk.
    """
    with store.exclusive():
        client.reset_practice()
        store.clear()
