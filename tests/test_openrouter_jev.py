"""OpenRouter transport and Jev parsing against a fake server. No real network calls.

Response bodies follow the documented shapes (OpenRouter Decisions API reference,
chat completions with usage.cost).
"""

from __future__ import annotations

import json

import httpx
import pytest
from conftest import jev_answers

from escal8r.config import DEEPSEEK_MODEL, JEV_MODEL
from escal8r.jev import DECISION_QUESTIONS, VERIFY_QUESTIONS, Jev
from escal8r.openrouter import (
    BudgetExceeded,
    CostTracker,
    ModelError,
    OpenRouterClient,
    SetupError,
    parse_json_text,
)


class Server:
    def __init__(self, *responses: httpx.Response):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


def client_for(server: Server, limit: float = 1.0) -> OpenRouterClient:
    return OpenRouterClient(
        "sk-test",
        CostTracker(limit_usd=limit),
        transport=httpx.MockTransport(server),
        sleep=lambda _s: None,
    )


def decisions_response(answers: dict, cost: float | None = 0.00002) -> httpx.Response:
    usage: dict = {"input_tokens": 476, "output_tokens": 70}
    if cost is not None:
        usage["cost"] = cost
    return httpx.Response(
        200,
        json={
            "id": "gen-dec-1",
            "model": f"{JEV_MODEL}-20260917",
            "provider": "TypeSafe",
            "answers": answers,
            "usage": usage,
        },
    )


def chat_response(content: str, cost: float | None = 0.0001) -> httpx.Response:
    usage: dict = {"prompt_tokens": 1000, "completion_tokens": 200}
    if cost is not None:
        usage["cost"] = cost
    return httpx.Response(
        200,
        json={
            "model": DEEPSEEK_MODEL,
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": usage,
        },
    )


# -- Jev ---------------------------------------------------------------------


def test_jev_decide_request_shape_and_parsing():
    server = Server(decisions_response(jev_answers(language="de", refund_or_disputed_charge=0.7)))
    with client_for(server) as client:
        decision = Jev(client).decide({"ticket": {"subject": "s", "body": "b"}})

    request = server.requests[0]
    assert request.url.path == "/api/alpha/decisions"
    assert request.headers["Authorization"] == "Bearer sk-test"
    body = server.body()
    assert body["model"] == JEV_MODEL
    assert body["provider"]["data_collection"] == "deny"
    assert "reading_rules" in body["state"]
    assert set(body["questions"]) == set(DECISION_QUESTIONS)
    assert body["questions"]["refund_or_disputed_charge"]["criteria"].keys() == {"true", "false"}

    assert decision.language_code == "de"
    assert decision.gates["refund_or_disputed_charge"] == 0.7
    assert decision.min_confidence == 0.9
    assert client.tracker.calls[0].model_served == f"{JEV_MODEL}-20260917"
    assert client.tracker.calls[0].cost_reported


def test_jev_verify_parses_nouls():
    answers = {name: {"type": "noul", "noul": 0.9} for name in VERIFY_QUESTIONS}
    server = Server(decisions_response(answers))
    with client_for(server) as client:
        assert Jev(client).verify_reply({"draft_reply": "x"}) == dict.fromkeys(
            VERIFY_QUESTIONS, 0.9
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.pop("spam"),
        lambda a: a.update(spam={"type": "refusal"}),
        lambda a: a.update(spam={"type": "noul", "noul": 1.7}),
        lambda a: a["language"].update(choice="fr"),
        lambda a: a["category"].update(type="score"),
    ],
)
def test_jev_bad_answers_raise_model_error(mutate):
    answers = jev_answers()
    mutate(answers)
    server = Server(decisions_response(answers))
    with client_for(server) as client, pytest.raises(ModelError):
        Jev(client).decide({})


def test_missing_cost_is_estimated_from_list_price():
    server = Server(decisions_response(jev_answers(), cost=None))
    with client_for(server) as client:
        Jev(client).decide({})
    call = client.tracker.calls[0]
    assert not call.cost_reported
    assert call.cost_usd == pytest.approx(476 * 0.042 / 1_000_000)


# -- transport -----------------------------------------------------------------


def test_chat_json_request_and_parse():
    server = Server(chat_response('```json\n{"draft_reply": "Hoi"}\n```'))
    with client_for(server) as client:
        data = client.chat_json(DEEPSEEK_MODEL, "sys", "user", purpose="t", max_tokens=100)
    assert data == {"draft_reply": "Hoi"}
    body = server.body()
    assert server.requests[0].url.path == "/api/v1/chat/completions"
    assert body["model"] == DEEPSEEK_MODEL
    assert body["provider"]["data_collection"] == "deny"
    assert body["reasoning"] == {"enabled": False}
    assert body["response_format"] == {"type": "json_object"}


def test_retries_then_succeeds():
    server = Server(
        httpx.Response(429, json={"error": {"code": 429, "message": "Rate limit exceeded"}}),
        httpx.Response(503, json={"error": {"code": 503, "message": "unavailable"}}),
        chat_response('{"ok": true}'),
    )
    with client_for(server) as client:
        assert client.chat_json(DEEPSEEK_MODEL, "s", "u", purpose="t", max_tokens=10) == {
            "ok": True
        }
    assert len(server.requests) == 3


def test_gives_up_after_max_attempts():
    server = Server(*[httpx.Response(502, json={"error": {"message": "x"}})] * 3)
    with client_for(server) as client, pytest.raises(ModelError, match="after 3 attempts"):
        client.chat_json(DEEPSEEK_MODEL, "s", "u", purpose="t", max_tokens=10)


def test_non_retryable_error_fails_fast():
    server = Server(httpx.Response(422, json={"error": {"message": "Unprocessable"}}))
    with client_for(server) as client, pytest.raises(ModelError, match="Unprocessable"):
        client.chat_json(DEEPSEEK_MODEL, "s", "u", purpose="t", max_tokens=10)
    assert len(server.requests) == 1


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_setup_errors_stop_the_run_instead_of_failing_one_ticket(status):
    error = {"error": {"code": status, "message": "No endpoints found matching your data policy"}}
    server = Server(httpx.Response(status, json=error))
    with client_for(server) as client, pytest.raises(SetupError, match="data policy"):
        Jev(client).decide({})
    assert len(server.requests) == 1
    assert not issubclass(SetupError, ModelError)  # the agent must not swallow it


def test_malformed_json_is_retried_once():
    server = Server(chat_response("not json"), chat_response('{"ok": true}'))
    with client_for(server) as client:
        assert client.chat_json(DEEPSEEK_MODEL, "s", "u", purpose="t", max_tokens=10) == {
            "ok": True
        }
        assert len(client.tracker.calls) == 2  # both attempts are paid for and counted


def test_invalid_json_content_raises():
    server = Server(*[chat_response("Sure! Here is the reply: Hoi")] * 2)
    with client_for(server) as client, pytest.raises(ModelError):
        client.chat_json(DEEPSEEK_MODEL, "s", "u", purpose="t", max_tokens=10)


def test_budget_blocks_the_next_call():
    server = Server(chat_response('{"a": 1}', cost=0.02), chat_response('{"a": 2}'))
    with client_for(server, limit=0.01) as client:
        client.chat_json(DEEPSEEK_MODEL, "s", "u", purpose="t", max_tokens=10)
        with pytest.raises(BudgetExceeded):
            client.chat_json(DEEPSEEK_MODEL, "s", "u", purpose="t", max_tokens=10)
    assert len(server.requests) == 1


def test_parse_json_text_rejects_non_objects():
    with pytest.raises(ModelError):
        parse_json_text("[1, 2]")
