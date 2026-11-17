"""Runner behaviour: one answer per ticket, crash recovery, dry-run. No network."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from escal8r.helpdesk import AlreadyHandled, HelpdeskError, OutcomeUnknown, RetriesExhausted
from escal8r.models import Decided, Decision, Ticket
from escal8r.runner import reset_all, run
from escal8r.state import AlreadyRunning, StateStore, Status


class Crash(BaseException):
    """Stands in for the process dying (kill -9, power loss) mid-post."""


class FakeHelpdesk:
    """In-memory helpdesk that, like the real one, refuses a second answer."""

    def __init__(self, ids: list[str]):
        self.tickets = {tid: Ticket(id=tid, subject="s", body="b") for tid in ids}
        self.answers: dict[str, list[Decision]] = {}
        self.failures: dict[str, list[BaseException]] = {}  # raised before saving
        self.lose_response: set[str] = set()  # saved, but the caller never hears back
        self.crash_after_save: set[str] = set()
        self.resets = 0

    def iter_tickets(self, status: str = "open") -> Iterator[Ticket]:
        yield from (t for tid, t in self.tickets.items() if tid not in self.answers)

    def post_decision(self, ticket_id: str, decision: Decision) -> dict[str, Any]:
        queued = self.failures.get(ticket_id)
        if queued:
            raise queued.pop(0)
        if ticket_id in self.answers:
            raise AlreadyHandled("POST", ticket_id, 409, "already_handled")
        self.answers[ticket_id] = [decision]
        if ticket_id in self.lose_response:
            self.lose_response.discard(ticket_id)
            raise OutcomeUnknown("read timeout")
        if ticket_id in self.crash_after_save:
            self.crash_after_save.discard(ticket_id)
            raise Crash
        return {"ok": True}

    def reset_practice(self) -> dict[str, Any]:
        self.resets += 1
        self.answers.clear()
        return {}


class CountingDecider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, ticket: Ticket) -> Decided:
        self.calls.append(ticket.id)
        decision = Decision(
            action="close", language="nl", category="other", priority="P4", reason="test"
        )
        return Decided(decision=decision, trace={"rule": "test"})


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "state.json")


def test_posts_each_open_ticket_once(store):
    desk = FakeHelpdesk(["T001", "T002", "T003"])
    summary = run(desk, store, CountingDecider())
    assert summary.posted == ["T001", "T002", "T003"]
    assert store.counts()["done"] == 3


def test_second_run_does_nothing(store):
    desk, decide = FakeHelpdesk(["T001", "T002"]), CountingDecider()
    run(desk, store, decide)
    summary = run(desk, store, decide)
    assert summary.counts()["posted"] == 0
    assert decide.calls == ["T001", "T002"]  # no re-deciding either


def test_409_marks_done_without_raising(store):
    desk = FakeHelpdesk(["T001"])
    desk.failures["T001"] = [AlreadyHandled("POST", "/x", 409, "already_handled")]
    summary = run(desk, store, CountingDecider())
    assert summary.already_handled == ["T001"]
    assert store.get("T001").status == Status.DONE


def test_lost_response_then_rerun_does_not_double_post(store):
    desk, decide = FakeHelpdesk(["T001"]), CountingDecider()
    desk.lose_response.add("T001")

    first = run(desk, store, decide)
    assert first.outcome_unknown == ["T001"]
    assert store.get("T001").status == Status.IN_PROGRESS

    second = run(desk, store, decide)
    assert second.recovered == ["T001"]
    assert second.counts()["posted"] == 0
    assert len(desk.answers["T001"]) == 1


def test_crash_after_save_then_rerun_does_not_double_post(tmp_path):
    path = tmp_path / "state.json"
    desk, decide = FakeHelpdesk(["T001", "T002"]), CountingDecider()
    desk.crash_after_save.add("T001")

    with pytest.raises(Crash):
        run(desk, StateStore(path), decide)

    summary = run(desk, StateStore(path), decide)  # fresh process, state from disk
    assert summary.recovered == ["T001"]
    assert summary.posted == ["T002"]
    assert all(len(answers) == 1 for answers in desk.answers.values())


def test_crash_before_save_resends_stored_decision_without_redeciding(tmp_path):
    path = tmp_path / "state.json"
    desk, decide = FakeHelpdesk(["T001"]), CountingDecider()
    desk.failures["T001"] = [Crash()]

    with pytest.raises(Crash):
        run(desk, StateStore(path), decide)
    assert StateStore(path).get("T001").status == Status.IN_PROGRESS

    summary = run(desk, StateStore(path), decide)
    assert summary.posted == ["T001"]
    assert decide.calls == ["T001"]  # decided once, sent from state the second time


def test_retries_exhausted_marks_failed_and_next_run_retries(store):
    desk = FakeHelpdesk(["T001"])
    desk.failures["T001"] = [RetriesExhausted("503 x6")]
    assert run(desk, store, CountingDecider()).failed == ["T001"]
    assert store.get("T001").status == Status.FAILED
    assert run(desk, store, CountingDecider()).posted == ["T001"]


def test_validation_error_marks_failed(store):
    desk = FakeHelpdesk(["T001"])
    desk.failures["T001"] = [HelpdeskError("POST", "/x", 422, "validation_failed")]
    assert run(desk, store, CountingDecider()).failed == ["T001"]


def test_decider_error_fails_one_ticket_not_the_run(store):
    desk = FakeHelpdesk(["T001", "T002"])

    def decide(ticket: Ticket) -> Decided:
        if ticket.id == "T001":
            raise RuntimeError("model down")
        return CountingDecider()(ticket)

    summary = run(desk, store, decide)
    assert summary.failed == ["T001"]
    assert summary.posted == ["T002"]


def test_done_locally_but_open_remotely_is_skipped(store):
    desk = FakeHelpdesk(["T001"])
    run(desk, store, CountingDecider())
    desk.answers.clear()  # helpdesk reset behind our back
    summary = run(desk, store, CountingDecider())
    assert summary.skipped_done == ["T001"]
    assert desk.answers == {}


def test_dry_run_posts_and_saves_nothing(tmp_path):
    path = tmp_path / "state.json"
    desk = FakeHelpdesk(["T001", "T002"])
    summary = run(desk, StateStore(path), CountingDecider(), dry_run=True)
    assert set(summary.dry_run) == {"T001", "T002"}
    assert desk.answers == {}
    assert not path.exists()


def test_reset_all_reopens_and_clears(store):
    desk = FakeHelpdesk(["T001"])
    run(desk, store, CountingDecider())
    reset_all(desk, store)
    assert desk.resets == 1
    assert store.counts()["done"] == 0
    assert run(desk, store, CountingDecider()).posted == ["T001"]


def test_concurrent_run_is_refused(tmp_path):
    path = tmp_path / "state.json"
    desk = FakeHelpdesk(["T001"])
    with StateStore(path).exclusive(), pytest.raises(AlreadyRunning):
        run(desk, StateStore(path), CountingDecider())
    assert desk.answers == {}


def test_lock_is_released_after_a_crash(tmp_path):
    path = tmp_path / "state.json"
    desk = FakeHelpdesk(["T001"])
    desk.failures["T001"] = [Crash()]
    with pytest.raises(Crash):
        run(desk, StateStore(path), CountingDecider())
    assert run(desk, StateStore(path), CountingDecider()).posted == ["T001"]


def test_reset_all_keeps_state_if_api_reset_fails(store):
    desk = FakeHelpdesk(["T001"])
    run(desk, store, CountingDecider())

    def broken_reset() -> dict[str, Any]:
        raise RetriesExhausted("503")

    desk.reset_practice = broken_reset  # type: ignore[method-assign]
    with pytest.raises(RetriesExhausted):
        reset_all(desk, store)
    assert store.counts()["done"] == 1


def test_budget_stop_leaves_remaining_tickets_open(store):
    from escal8r.openrouter import BudgetExceeded

    desk, counting = FakeHelpdesk(["T001", "T002", "T003"]), CountingDecider()

    def decide(ticket: Ticket) -> Decided:
        if ticket.id == "T002":
            raise BudgetExceeded("limit reached")
        return counting(ticket)

    summary = run(desk, store, decide)
    assert summary.posted == ["T001"]
    assert summary.not_started == ["T002", "T003"]
    assert store.get("T002") is None  # untouched, so the next run picks it up


def test_one_log_line_per_ticket(tmp_path):
    import json

    from escal8r.ticketlog import TicketLog

    desk = FakeHelpdesk(["T001", "T002", "T003"])
    desk.failures["T002"] = [AlreadyHandled("POST", "/x", 409, "already_handled")]
    ticket_log = TicketLog(tmp_path / "logs", run_id="t")
    run(desk, StateStore(tmp_path / "state.json"), CountingDecider(), ticket_log=ticket_log)

    lines = [json.loads(line) for line in ticket_log.path.read_text().splitlines()]
    assert [(line["ticket"], line["outcome"]) for line in lines] == [
        ("T001", "posted"),
        ("T002", "already_handled"),
        ("T003", "posted"),
    ]
    assert lines[0]["trace"] == {"rule": "test"}


def test_setup_error_posts_nothing_and_reports_why(store):
    from escal8r.openrouter import SetupError

    desk = FakeHelpdesk(["T001", "T002"])

    def decide(ticket: Ticket) -> Decided:
        raise SetupError("OpenRouter rejected /alpha/decisions: HTTP 404 model not found")

    summary = run(desk, store, decide)
    assert desk.answers == {}
    assert summary.not_started == ["T001", "T002"]
    assert "404" in summary.stopped
    assert store.counts() == {"in_progress": 0, "done": 0, "failed": 0}


def test_limit_handles_first_n_and_next_run_continues(store):
    desk, decide = FakeHelpdesk(["T001", "T002", "T003", "T004"]), CountingDecider()
    assert run(desk, store, decide, limit=2).posted == ["T001", "T002"]
    assert run(desk, store, decide, limit=2).posted == ["T003", "T004"]
    assert decide.calls == ["T001", "T002", "T003", "T004"]


def test_limit_does_not_hide_tickets_from_recovery(store):
    desk = FakeHelpdesk(["T001", "T002", "T003"])
    desk.lose_response.add("T003")
    run(desk, store, CountingDecider())  # T003 saved, response lost -> in_progress
    summary = run(desk, store, CountingDecider(), limit=1)
    assert summary.recovered == ["T003"]
