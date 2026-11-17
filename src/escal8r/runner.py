"""Processes open tickets: decide -> save in_progress -> post -> save done -> log.

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
from escal8r.models import Decided, Decision, Ticket
from escal8r.openrouter import StopRun
from escal8r.state import StateStore, Status
from escal8r.ticketlog import TicketLog

log = logging.getLogger(__name__)

DecideFn = Callable[[Ticket], Decided]


class Helpdesk(Protocol):
    def iter_tickets(self, status: str = "open") -> Iterator[Ticket]: ...
    def post_decision(self, ticket_id: str, decision: Decision) -> dict[str, Any]: ...
    def reset_practice(self) -> dict[str, Any]: ...


class _NullLog:
    def write(self, ticket_id: str, outcome: str, *args: Any, **kwargs: Any) -> None:
        pass


@dataclass
class RunSummary:
    posted: list[str] = field(default_factory=list)
    already_handled: list[str] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)
    skipped_done: list[str] = field(default_factory=list)
    outcome_unknown: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    not_started: list[str] = field(default_factory=list)  # run stopped early
    dry_run: dict[str, Decision] = field(default_factory=dict)
    actions: dict[str, int] = field(default_factory=dict)
    stopped: str | None = None  # why the run stopped early, if it did

    def counts(self) -> dict[str, int]:
        return {
            name: len(value)
            for name, value in vars(self).items()
            if name not in ("actions", "stopped")
        }


def run(
    client: Helpdesk,
    store: StateStore,
    decide: DecideFn,
    *,
    ticket_log: TicketLog | None = None,
    dry_run: bool = False,
) -> RunSummary:
    """Handle every open ticket once. In dry-run mode nothing is posted or saved."""
    logger: TicketLog | _NullLog = ticket_log or _NullLog()
    if dry_run:
        return _run(client, store, decide, logger, dry_run=True)
    with store.exclusive():
        return _run(client, store, decide, logger, dry_run=False)


def _run(
    client: Helpdesk,
    store: StateStore,
    decide: DecideFn,
    logger: TicketLog | _NullLog,
    *,
    dry_run: bool,
) -> RunSummary:
    summary = RunSummary()
    open_tickets = list(client.iter_tickets("open"))
    log.info("%d open tickets", len(open_tickets))

    if not dry_run:
        summary.recovered = store.recover({t.id for t in open_tickets})
        for ticket_id in summary.recovered:
            log.info("%s: recovered, helpdesk already has the decision", ticket_id)
            logger.write(ticket_id, "recovered")

    for index, ticket in enumerate(open_tickets):
        entry = store.get(ticket.id)
        if entry and entry.status == Status.DONE:
            # Open in the helpdesk but done locally: the helpdesk was reset without
            # clearing local state. Never risk a second answer; `escal8r reset` clears both.
            log.warning("%s: done locally but open in helpdesk; skipped", ticket.id)
            summary.skipped_done.append(ticket.id)
            logger.write(ticket.id, "skipped_done_locally")
            continue

        try:
            if entry and entry.status == Status.IN_PROGRESS and entry.decision:
                decided = Decided(decision=entry.decision, trace={"resent_from_state": True})
            else:
                decided = decide(ticket)
        except StopRun as exc:
            log.error("stopping run: %s", exc)
            summary.stopped = str(exc)
            summary.not_started = [t.id for t in open_tickets[index:]]
            break
        except Exception as exc:
            log.exception("%s: decision failed", ticket.id)
            if not dry_run:
                store.mark_failed(ticket.id, note="decision failed")
            summary.failed.append(ticket.id)
            logger.write(ticket.id, "decision_failed", error=type(exc).__name__)
            continue

        action = decided.decision.action
        summary.actions[action] = summary.actions.get(action, 0) + 1
        if dry_run:
            summary.dry_run[ticket.id] = decided.decision
            logger.write(ticket.id, "dry_run", decided.decision, decided.trace)
            continue

        outcome, error = _post(client, store, ticket.id, decided.decision, summary)
        logger.write(ticket.id, outcome, decided.decision, decided.trace, error)

    log.info("run finished: %s actions=%s", summary.counts(), summary.actions)
    return summary


def _post(
    client: Helpdesk,
    store: StateStore,
    ticket_id: str,
    decision: Decision,
    summary: RunSummary,
) -> tuple[str, str | None]:
    """Post one decision. Returns (outcome, error) for the log."""
    store.mark_in_progress(ticket_id, decision)
    try:
        client.post_decision(ticket_id, decision)
    except AlreadyHandled:
        store.mark_done(ticket_id, note="409 already handled")
        summary.already_handled.append(ticket_id)
        log.warning("%s: 409 already handled; marked done", ticket_id)
        return "already_handled", None
    except OutcomeUnknown as exc:
        # Stay in_progress: the next run checks whether the helpdesk saved it.
        summary.outcome_unknown.append(ticket_id)
        log.warning("%s: outcome unknown (%s); will verify next run", ticket_id, exc)
        return "outcome_unknown", str(exc)
    except (HelpdeskError, RetriesExhausted) as exc:
        # The helpdesk saved nothing; safe to try again on the next run.
        store.mark_failed(ticket_id, note=str(exc))
        summary.failed.append(ticket_id)
        log.error("%s: post failed: %s", ticket_id, exc)
        return "post_failed", str(exc)
    store.mark_done(ticket_id, note=f"posted {decision.action}")
    summary.posted.append(ticket_id)
    log.info("%s: posted %s", ticket_id, decision.action)
    return "posted", None


def reset_all(client: Helpdesk, store: StateStore) -> None:
    """Re-open all practice tickets, then clear local state.

    API first: if the reset fails, local state still matches the helpdesk.
    """
    with store.exclusive():
        client.reset_practice()
        store.clear()
