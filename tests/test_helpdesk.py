"""Helpdesk client behaviour against a fake server. No real network calls."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from escal8r.helpdesk import (
    AlreadyHandled,
    HelpdeskClient,
    HelpdeskError,
    OutcomeUnknown,
    RateLimiter,
    RetriesExhausted,
    backoff_delay,
)
from escal8r.models import Decision

Handler = Callable[[httpx.Request], httpx.Response]


class FakeServer:
    """Replays a scripted list of responses (or exceptions) and records requests."""

    def __init__(self, *script: httpx.Response | Exception | Handler):
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step) and not isinstance(step, httpx.Response):
            return step(request)
        return step


def make_client(server: FakeServer, sleeps: list[float] | None = None) -> HelpdeskClient:
    record = sleeps.append if sleeps is not None else (lambda _s: None)
    return HelpdeskClient(
        "https://helpdesk.test/v1",
        "test-key",
        "me@example.com",
        "Test Person",
        transport=httpx.MockTransport(server),
        limiter=RateLimiter(1000, sleep=lambda _s: None),
        sleep=record,
        max_attempts=4,
    )


def ok(body: object) -> httpx.Response:
    return httpx.Response(200, json=body)


def err(status: int, error: str, **headers: str) -> httpx.Response:
    return httpx.Response(status, json={"error": error}, headers=headers)


REPLY = Decision(
    action="reply",
    language="nl",
    category="question",
    priority="P4",
    reason="How-to answered by F02.",
    draft_reply="Hoi, ga naar Team.",
)


def test_every_request_carries_auth_and_email():
    server = FakeServer(ok({"mode": "practice"}))
    with make_client(server) as client:
        client.status()
    headers = server.requests[0].headers
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["X-Candidate-Email"] == "me@example.com"
    assert headers["X-Candidate-Name"] == "Test Person"


def test_503_is_retried_with_backoff():
    server = FakeServer(err(503, "service_unavailable"), err(503, "service_unavailable"), ok({}))
    sleeps: list[float] = []
    with make_client(server, sleeps) as client:
        client.status()
    assert len(server.requests) == 3
    assert len(sleeps) == 2
    assert 0.25 <= sleeps[0] <= 0.5 and 0.5 <= sleeps[1] <= 1.0


def test_503_on_post_is_retried_because_nothing_was_saved():
    server = FakeServer(err(503, "service_unavailable"), ok({"ok": True}))
    with make_client(server) as client:
        assert client.post_decision("T005", REPLY) == {"ok": True}
    assert len(server.requests) == 2


def test_429_waits_retry_after():
    server = FakeServer(err(429, "rate_limited", **{"Retry-After": "2"}), ok({}))
    sleeps: list[float] = []
    with make_client(server, sleeps) as client:
        client.status()
    assert sleeps == [2.0]


def test_409_raises_already_handled_without_retry():
    server = FakeServer(err(409, "already_handled"))
    with make_client(server) as client, pytest.raises(AlreadyHandled):
        client.post_decision("T005", REPLY)
    assert len(server.requests) == 1


def test_422_raises_with_problems():
    response = httpx.Response(
        422, json={"error": "validation_failed", "problems": ["draft_reply missing"]}
    )
    server = FakeServer(response)
    with make_client(server) as client, pytest.raises(HelpdeskError) as info:
        client.post_decision("T005", REPLY)
    assert info.value.status == 422
    assert info.value.detail == ["draft_reply missing"]


def test_post_read_timeout_is_outcome_unknown_not_retried():
    server = FakeServer(httpx.ReadTimeout("lost"))
    with make_client(server) as client, pytest.raises(OutcomeUnknown):
        client.post_decision("T005", REPLY)
    assert len(server.requests) == 1


def test_post_500_is_outcome_unknown():
    server = FakeServer(err(500, "internal"))
    with make_client(server) as client, pytest.raises(OutcomeUnknown):
        client.post_decision("T005", REPLY)


def test_connect_error_is_retried_even_for_post():
    server = FakeServer(httpx.ConnectError("refused"), ok({"ok": True}))
    with make_client(server) as client:
        client.post_decision("T005", REPLY)
    assert len(server.requests) == 2


def test_get_read_timeout_is_retried():
    server = FakeServer(httpx.ReadTimeout("slow"), ok({}))
    with make_client(server) as client:
        client.status()
    assert len(server.requests) == 2


def test_retries_exhausted():
    server = FakeServer(*[err(503, "service_unavailable")] * 4)
    with make_client(server) as client, pytest.raises(RetriesExhausted):
        client.status()


def test_post_decision_hits_action_endpoint_with_payload():
    server = FakeServer(ok({}))
    with make_client(server) as client:
        client.post_decision("T005", REPLY)
    request = server.requests[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/tickets/T005/reply"
    body = json.loads(request.content)
    assert "action" not in body
    assert body["draft_reply"] == "Hoi, ga naar Team."


def test_iter_tickets_follows_cursor():
    def page(tickets: list[str], cursor: str | None) -> httpx.Response:
        return ok({"tickets": [{"id": t, "body": None} for t in tickets], "next_cursor": cursor})

    server = FakeServer(page(["T001", "T002"], "T002"), page(["T003"], None))
    with make_client(server) as client:
        ids = [t.id for t in client.iter_tickets()]
    assert ids == ["T001", "T002", "T003"]
    assert server.requests[0].url.params["status"] == "open"
    assert "cursor" not in server.requests[0].url.params
    assert server.requests[1].url.params["cursor"] == "T002"


def test_examples_parse_labels():
    example = {
        "id": "E01",
        "subject": "s",
        "body": "b",
        "label": {
            "language": "en",
            "category": "bug",
            "priority": "P3",
            "action": "reply",
            "reason": "r",
        },
    }
    server = FakeServer(ok({"examples": [example]}))
    with make_client(server) as client:
        [parsed] = client.examples()
    assert parsed.label.action == "reply"


def test_rate_limiter_spaces_calls():
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(4, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        limiter.wait()
    assert sleeps == pytest.approx([0.25, 0.25])


def test_backoff_is_capped():
    assert backoff_delay(20, rng=lambda: 1.0) == 8.0
    assert backoff_delay(0, rng=lambda: 0.0) == 0.25
