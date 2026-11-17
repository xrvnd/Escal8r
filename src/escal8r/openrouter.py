"""Shared OpenRouter transport for both models: retries, cost tracking, budget guard.

- Chat completions (DeepSeek) return JSON text that we parse.
- The alpha Decisions endpoint (Jev) returns typed answers; see jev.py.
Every call records the model that actually served it and its cost.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from escal8r import tracing
from escal8r.config import (
    ALLOWED_MODELS,
    MODEL_PRICES_PER_M,
    OPENROUTER_BASE_URL,
    PROVIDER_PREFERENCES,
)
from escal8r.helpdesk import Sleep, backoff_delay
from escal8r.tracing import current_span

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 524, 529}
# Our request shapes are fixed, so these mean a broken setup (bad key, no credits, unknown
# model, rejected parameter or provider policy), not one bad ticket: stop the whole run
# instead of quietly handing off every ticket.
SETUP_ERROR_STATUS = {400, 401, 402, 403, 404}
DEFAULT_MAX_ATTEMPTS = 3
JSON_ATTEMPTS = 2  # one retry when a model returns malformed JSON
DEFAULT_TIMEOUT_S = 90.0


class ModelError(RuntimeError):
    """A model call failed after retries, or returned something unusable."""


class StopRun(RuntimeError):
    """Stop processing further tickets (the remaining ones stay open)."""


class BudgetExceeded(StopRun):
    pass


class SetupError(StopRun):
    """OpenRouter rejected the request itself (key, credits, model id, parameters)."""


@dataclass(frozen=True)
class CallRecord:
    purpose: str
    model_requested: str
    model_served: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_reported: bool  # False: estimated from list prices


@dataclass
class CostTracker:
    limit_usd: float
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def spent_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    def check(self) -> None:
        if self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(
                f"model spend ${self.spent_usd:.4f} reached the ${self.limit_usd:.2f} limit"
            )

    def since(self, index: int) -> list[CallRecord]:
        return self.calls[index:]


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = MODEL_PRICES_PER_M.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def parse_json_text(text: str) -> dict[str, Any]:
    """Parse a model's JSON answer, tolerating a ```json fence around it."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ModelError(f"model did not return valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelError("model JSON is not an object")
    return value


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        tracker: CostTracker,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Sleep = time.sleep,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self.tracker = tracker
        self._http = httpx.Client(
            base_url=OPENROUTER_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "Escal8r"},
            timeout=timeout_s,
            transport=transport,
        )
        self._sleep = sleep
        self._max_attempts = max_attempts

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- endpoints -------------------------------------------------------------

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        *,
        purpose: str,
        max_tokens: int,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """One chat completion whose answer must be a JSON object."""
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False},  # thinking tokens are billed as output
            "usage": {"include": True},
            "provider": {**PROVIDER_PREFERENCES, "sort": "price"},
        }
        prompt_version = hashlib.sha256(system.encode()).hexdigest()[:10]
        for attempt in range(JSON_ATTEMPTS):
            with tracing.span(
                "model_call", purpose=purpose, model_requested=model, prompt_version=prompt_version
            ) as span:
                tracing.record_text(span, "prompt", user)
                response = self._post("/v1/chat/completions", body)
                usage = response.get("usage") or {}
                self._record(
                    purpose,
                    model,
                    response,
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    reported_cost=usage.get("cost"),
                )
                try:
                    content = _message_content(response)
                    tracing.record_text(span, "output", content)
                    return parse_json_text(content)
                except ModelError as exc:
                    if attempt + 1 == JSON_ATTEMPTS:
                        raise
                    span.fail(str(exc))
                    log.warning("%s: %s; asking once more", purpose, exc)
        raise AssertionError("unreachable")

    def decide(
        self,
        model: str,
        state: Any,
        questions: dict[str, dict[str, Any]],
        *,
        purpose: str,
    ) -> dict[str, Any]:
        """One Decisions call. Returns the raw `answers` map."""
        body = {
            "model": model,
            "state": state,
            "questions": questions,
            "provider": dict(PROVIDER_PREFERENCES),
        }
        with tracing.span("model_call", purpose=purpose, model_requested=model) as span:
            tracing.record_text(span, "state", json.dumps(state, ensure_ascii=False))
            response = self._post("/alpha/decisions", body)
            usage = response.get("usage") or {}
            self._record(
                purpose,
                model,
                response,
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                reported_cost=usage.get("cost"),
            )
            answers = response.get("answers")
            if not isinstance(answers, dict):
                raise ModelError("decisions response has no answers map")
            span.set(answers=_compact_answers(answers))
            return answers

    def list_models(self) -> list[dict[str, Any]]:
        self.tracker.check()
        response = self._request("GET", "/v1/models", None)
        data = response.get("data")
        return data if isinstance(data, list) else []

    # -- internals -------------------------------------------------------------

    def _record(
        self,
        purpose: str,
        model: str,
        response: dict[str, Any],
        *,
        input_tokens: int,
        output_tokens: int,
        reported_cost: Any,
    ) -> None:
        reported = isinstance(reported_cost, int | float)
        cost = (
            float(reported_cost) if reported else _estimate_cost(model, input_tokens, output_tokens)
        )
        served = str(response.get("model") or model)
        self.tracker.calls.append(
            CallRecord(purpose, model, served, input_tokens, output_tokens, cost, reported)
        )
        span = current_span()
        if span is not None and span.kind == "model_call":
            span.set(
                model_served=served,
                generation_id=response.get("id"),
                provider=response.get("provider"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                cost_reported=reported,
            )
        if _undated(served) not in ALLOWED_MODELS:
            # OpenRouter or the provider rerouted us to a model we did not choose.
            raise SetupError(f"{purpose}: asked for {model}, served by {served}; stopping")

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        model = str(body.get("model"))
        if model not in ALLOWED_MODELS:
            raise SetupError(f"model {model!r} is not in ALLOWED_MODELS; refusing to call it")
        self.tracker.check()
        return self._request("POST", path, body)

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        last_error = ""
        attempts: list[Any] = []
        span = current_span()
        if span is not None and span.kind == "model_call":
            span.set(attempts=attempts)
        for attempt in range(self._max_attempts):
            try:
                response = self._http.request(method, path, json=body)
            except httpx.TransportError as exc:
                last_error = type(exc).__name__
                attempts.append(last_error)
            else:
                attempts.append(response.status_code)
                if response.status_code < 400:
                    try:
                        value = response.json()
                    except ValueError as exc:
                        raise ModelError(f"{path}: response is not JSON") from exc
                    if not isinstance(value, dict):
                        raise ModelError(f"{path}: response is not a JSON object")
                    return value
                last_error = f"HTTP {response.status_code} {_error_message(response)}"
                if response.status_code in SETUP_ERROR_STATUS:
                    raise SetupError(f"OpenRouter rejected {path}: {last_error}")
                if response.status_code not in RETRYABLE_STATUS:
                    raise ModelError(f"{path}: {last_error}")
            if attempt + 1 < self._max_attempts:
                delay = backoff_delay(attempt + 1)
                log.warning("%s: %s, retry in %.1fs", path, last_error, delay)
                self._sleep(delay)
        raise ModelError(f"{path}: failed after {self._max_attempts} attempts ({last_error})")


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        return ""
    return str(error.get("message", "")) if isinstance(error, dict) else str(error)


def _message_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError("chat completion has no message content") from exc
    if not isinstance(content, str):
        raise ModelError("chat completion content is not text")
    return content


def _undated(model_id: str) -> str:
    """'typesafe/jev-1.14-20260917' -> 'typesafe/jev-1.14': compare models without snapshot."""
    head, sep, tail = model_id.rpartition("-")
    return head if sep and tail.isdigit() and len(tail) == 8 else model_id


def _compact_answers(answers: dict[str, Any]) -> dict[str, Any]:
    """Jev answers as numbers only: choice -> [option, confidence], noul -> probability."""
    compact: dict[str, Any] = {}
    for name, answer in answers.items():
        if not isinstance(answer, dict):
            continue
        kind = answer.get("type")
        if kind == "choice":
            compact[name] = [answer.get("choice"), answer.get("confidence")]
        elif kind == "noul":
            compact[name] = answer.get("noul")
        else:
            compact[name] = kind
    return compact
