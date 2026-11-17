"""Crash-safe record of what the agent has done with each ticket.

Lifecycle per ticket:  (none) -> in_progress -> done
                                             -> failed  (retried on the next run)

`in_progress` is written *before* posting and holds the exact decision, so after a
crash the same decision can be re-sent without asking the models again.
Every change is written atomically (temp file + rename), so the file is never half-written.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from escal8r.models import Decision

STATE_VERSION = 1


class Status(StrEnum):
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class Entry(BaseModel):
    status: Status
    decision: Decision | None = None
    note: str = ""
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class _StateFile(BaseModel):
    version: int = STATE_VERSION
    tickets: dict[str, Entry] = Field(default_factory=dict)


class StateError(RuntimeError):
    """The state file exists but cannot be read. Refuse to guess."""


class AlreadyRunning(StateError):
    """Another process holds the run lock."""


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self._data = self._load()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold the run lock: one agent process at a time per state file.

        Two concurrent runs would each have their own rate limiter (exceeding the
        helpdesk limit) and race on the same tickets. The OS releases an `flock`
        when the process dies, even on kill -9, so a crash never leaves a stale lock.
        State is reloaded after locking, in case another run changed it.
        """
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AlreadyRunning(f"another escal8r run holds {lock_path}") from exc
            self._data = self._load()
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    # -- queries ---------------------------------------------------------------

    def get(self, ticket_id: str) -> Entry | None:
        return self._data.tickets.get(ticket_id)

    def ids_with_status(self, status: Status) -> list[str]:
        return sorted(tid for tid, entry in self._data.tickets.items() if entry.status == status)

    def counts(self) -> dict[str, int]:
        return {status.value: len(self.ids_with_status(status)) for status in Status}

    # -- transitions -----------------------------------------------------------

    def mark_in_progress(self, ticket_id: str, decision: Decision) -> None:
        self._set(ticket_id, Entry(status=Status.IN_PROGRESS, decision=decision))

    def mark_done(self, ticket_id: str, note: str) -> None:
        previous = self.get(ticket_id)
        decision = previous.decision if previous else None
        self._set(ticket_id, Entry(status=Status.DONE, decision=decision, note=note))

    def mark_failed(self, ticket_id: str, note: str) -> None:
        self._set(ticket_id, Entry(status=Status.FAILED, note=note))

    def recover(self, open_ids: set[str]) -> list[str]:
        """Resolve tickets left `in_progress` by a crash or a lost response.

        A ticket that is no longer open was saved by the helpdesk: mark it done.
        One that is still open was not saved: leave it `in_progress` so the runner
        re-sends the stored decision. Returns the ids marked done.
        """
        recovered = []
        for ticket_id in self.ids_with_status(Status.IN_PROGRESS):
            if ticket_id not in open_ids:
                self.mark_done(ticket_id, note="recovered: helpdesk already has it")
                recovered.append(ticket_id)
        return recovered

    def clear(self) -> None:
        self._data = _StateFile()
        self._save()

    # -- persistence -----------------------------------------------------------

    def _set(self, ticket_id: str, entry: Entry) -> None:
        self._data.tickets[ticket_id] = entry
        self._save()

    def _load(self) -> _StateFile:
        if not self.path.exists():
            return _StateFile()
        try:
            data = _StateFile.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError) as exc:
            raise StateError(
                f"Cannot read {self.path}: {exc}. Fix or remove it; starting fresh could "
                "re-process tickets."
            ) from exc
        if data.version != STATE_VERSION:
            raise StateError(f"{self.path} has version {data.version}, expected {STATE_VERSION}")
        return data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._data.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".state-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
