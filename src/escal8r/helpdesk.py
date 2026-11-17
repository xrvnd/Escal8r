"""Client for the Roosterly helpdesk API.

Reliability rules (see README > Reliability):
- every request carries the API key and candidate email;
- one global rate limiter keeps us under the 5 calls/s limit;
- 429: wait `Retry-After`, then retry;
- 503: exponential backoff with jitter, then retry (the API saved nothing);
- 409: raise `AlreadyHandled`, never retry;
- a POST whose response was lost (read timeout, 5xx other than 503) raises
  `OutcomeUnknown`: it may have been saved, so the caller must check before re-posting.
"""

from __future__ import annotations

import json as _json
import logging
import random
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import quote

import httpx

from escal8r import tracing
from escal8r.config import HELPDESK_CALLS_PER_SECOND, Settings
from escal8r.models import Action, Decision, Example, Ticket

log = logging.getLogger(__name__)

Sleep = Callable[[float], None]

PAGE_SIZE = 10  # API maximum
DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_TIMEOUT_S = 30.0
BACKOFF_BASE_S = 0.5
BACKOFF_CAP_S = 8.0
DEFAULT_RETRY_AFTER_S = 1.0


class HelpdeskError(RuntimeError):
    """The helpdesk refused a request in a way that retrying will not fix."""

    def __init__(self, method: str, path: str, status: int, error: str, detail: Any = None):
        super().__init__(f"{method} {path} -> HTTP {status} {error}")
        self.status = status
        self.error = error
        self.detail = detail


class AlreadyHandled(HelpdeskError):
    """409: the ticket already has a decision. Mark it done; never re-post."""


class OutcomeUnknown(RuntimeError):
    """A POST may or may not have been saved. Check the ticket before acting again."""


class RetriesExhausted(RuntimeError):
    """A retryable failure kept happening until we ran out of attempts."""


class RateLimiter:
    """Spaces calls at least `1 / calls_per_second` apart, across all threads."""

    def __init__(
        self,
        calls_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = time.sleep,
    ):
        if calls_per_second <= 0:
            raise ValueError("calls_per_second must be positive")
        self._interval = 1.0 / calls_per_second
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self) -> float:
        """Block until the next slot. Returns the seconds spent waiting."""
        with self._lock:
            now = self._clock()
            delay = max(0.0, self._next_slot - now)
            if delay > 0:
                self._sleep(delay)
                now += delay
            self._next_slot = now + self._interval
            return delay


def backoff_delay(attempt: int, rng: Callable[[], float] = random.random) -> float:
    """Exponential backoff with 'equal jitter': half fixed, half random.

    attempt 0 -> 0.25-0.5 s, 1 -> 0.5-1 s, 2 -> 1-2 s ... capped at 8 s.
    """
    ceiling = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2.0**attempt)
    return ceiling / 2 + rng() * ceiling / 2


def _retry_after_seconds(response: httpx.Response) -> float:
    try:
        return max(0.0, float(response.headers.get("Retry-After", DEFAULT_RETRY_AFTER_S)))
    except ValueError:
        return DEFAULT_RETRY_AFTER_S


def _error_code(response: httpx.Response) -> tuple[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return response.reason_phrase or "error", None
    if isinstance(body, dict):
        return str(body.get("error", "error")), body.get("problems") or body.get("message")
    return "error", body


class HelpdeskClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        candidate_email: str,
        candidate_name: str | None = None,
        *,
        limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Sleep = time.sleep,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Candidate-Email": candidate_email,
            "Accept": "application/json",
        }
        if candidate_name:
            headers["X-Candidate-Name"] = candidate_name
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )
        self._limiter = limiter or RateLimiter(HELPDESK_CALLS_PER_SECOND)
        self._sleep = sleep
        self._max_attempts = max_attempts

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> HelpdeskClient:
        return cls(
            settings.helpdesk_url,
            settings.helpdesk_key,
            settings.candidate_email,
            settings.candidate_name,
            **kwargs,
        )

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> HelpdeskClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- endpoints -------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return self._request_object("GET", "/status")

    def examples(self) -> list[Example]:
        return [Example.model_validate(item) for item in _items(self._request("GET", "/examples"))]

    def iter_tickets(self, status: str = "open") -> Iterator[Ticket]:
        """Yield every ticket in the current set, following `next_cursor`."""
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": PAGE_SIZE, "status": status}
            if cursor:
                params["cursor"] = cursor
            page = self._request_object("GET", "/tickets", params=params)
            for item in _items(page):
                yield Ticket.model_validate(item)
            cursor = page.get("next_cursor")
            if not cursor:
                return

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """One ticket plus how it was handled, if it was."""
        return self._request_object("GET", f"/tickets/{quote(ticket_id, safe='')}")

    def post_decision(self, ticket_id: str, decision: Decision) -> dict[str, Any]:
        action: Action = decision.action
        path = f"/tickets/{quote(ticket_id, safe='')}/{action}"
        return self._request_object("POST", path, json=decision.payload())

    def reset_practice(self) -> dict[str, Any]:
        return self._request_object("POST", "/practice/reset")

    # -- transport -------------------------------------------------------------

    def _request_object(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Like `_request`, but the response must be a JSON object."""
        body = self._request(method, path, **kwargs)
        if not isinstance(body, dict):
            raise ValueError(f"{method} {path}: expected a JSON object, got {type(body).__name__}")
        return body

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        with tracing.span("helpdesk", method=method, path=path) as span:
            if json is not None:
                tracing.record_text(span, "request", _json.dumps(json, ensure_ascii=False))
            attempts: list[dict[str, Any]] = []
            span.set(attempts=attempts)
            try:
                return self._send(method, path, params, json, attempts)
            finally:
                span.set(limiter_wait_s=round(sum(a.get("limiter_wait_s", 0) for a in attempts), 3))

    def _send(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        json: dict[str, Any] | None,
        attempts: list[dict[str, Any]],
    ) -> Any:
        is_post = method == "POST"

        def retry(result: str, delay: float) -> None:
            attempts[-1].update(result=result, retry_in_s=round(delay, 2))
            log.warning("%s %s: %s, retry in %.1fs", method, path, result, delay)
            self._sleep(delay)

        for attempt in range(self._max_attempts):
            attempts.append({"limiter_wait_s": round(self._limiter.wait(), 3)})
            try:
                response = self._http.request(method, path, params=params, json=json)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # The request never reached the server: always safe to retry.
                retry(type(exc).__name__, backoff_delay(attempt))
                continue
            except httpx.TransportError as exc:
                # Sent, but the response was lost.
                attempts[-1]["result"] = type(exc).__name__
                if is_post:
                    raise OutcomeUnknown(f"{method} {path}: {type(exc).__name__}") from exc
                retry(type(exc).__name__, backoff_delay(attempt))
                continue

            code = response.status_code
            attempts[-1]["result"] = code
            if code < 400:
                return response.json() if response.content else {}
            if code == 429:
                retry("429 rate limited", _retry_after_seconds(response))
                continue
            if code == 503:
                retry("503", backoff_delay(attempt))
                continue

            error, detail = _error_code(response)
            if code == 409:
                raise AlreadyHandled(method, path, code, error, detail)
            if code >= 500:
                if is_post:
                    # Unlike 503, other 5xx do not promise that nothing was saved.
                    raise OutcomeUnknown(f"{method} {path}: HTTP {code}")
                retry(f"HTTP {code}", backoff_delay(attempt))
                continue
            raise HelpdeskError(method, path, code, error, detail)

        raise RetriesExhausted(f"{method} {path} failed after {self._max_attempts} attempts")


def _items(payload: Any) -> list[dict[str, Any]]:
    """List of records from a response that is either a list or a wrapped list."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("tickets", "examples", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unexpected response shape: {type(payload).__name__}")
