import json

import pytest

from escal8r.models import Decision
from escal8r.state import StateError, StateStore, Status

DECISION = Decision(
    action="handoff",
    language="nl",
    category="billing",
    priority="P2",
    reason="Refund request.",
    handoff_note="Customer asks for a refund for August.",
)


def test_round_trip_survives_reload(tmp_path):
    path = tmp_path / "state.json"
    store = StateStore(path)
    store.mark_in_progress("T003", DECISION)

    reloaded = StateStore(path)
    entry = reloaded.get("T003")
    assert entry is not None
    assert entry.status == Status.IN_PROGRESS
    assert entry.decision == DECISION


def test_done_keeps_decision(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.mark_in_progress("T003", DECISION)
    store.mark_done("T003", note="posted")
    entry = store.get("T003")
    assert entry is not None and entry.status == Status.DONE and entry.decision == DECISION


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    store = StateStore(tmp_path / "state.json")
    for i in range(5):
        store.mark_done(f"T{i:03d}", note="posted")
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
    assert len(json.loads((tmp_path / "state.json").read_text())["tickets"]) == 5


def test_corrupt_file_is_refused_not_overwritten(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ not json")
    with pytest.raises(StateError):
        StateStore(path)
    assert path.read_text() == "{ not json"


def test_wrong_version_is_refused(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 99, "tickets": {}}))
    with pytest.raises(StateError):
        StateStore(path)


def test_recover_marks_only_closed_tickets_done(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.mark_in_progress("T001", DECISION)  # helpdesk saved it before the crash
    store.mark_in_progress("T002", DECISION)  # helpdesk did not

    recovered = store.recover(open_ids={"T002"})

    assert recovered == ["T001"]
    assert store.get("T001").status == Status.DONE
    assert store.get("T002").status == Status.IN_PROGRESS


def test_clear(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.mark_done("T001", note="posted")
    store.clear()
    assert StateStore(tmp_path / "state.json").counts() == {
        "in_progress": 0,
        "done": 0,
        "failed": 0,
    }
