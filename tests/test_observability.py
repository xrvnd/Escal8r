"""Tracing + the observability eval, end to end with only the network faked."""

from __future__ import annotations

import json
import re

import httpx
import pytest
from conftest import jev_answers

from escal8r.agent import Agent
from escal8r.config import FAQ_PATH, Thresholds
from escal8r.deepseek import DeepSeek
from escal8r.faq import load_faq
from escal8r.helpdesk import HelpdeskClient, RateLimiter
from escal8r.jev import VERIFY_QUESTIONS, Jev
from escal8r.observability import Trail, check_run, format_check, load_trail, render_trail
from escal8r.openrouter import CostTracker, OpenRouterClient
from escal8r.runner import run
from escal8r.state import StateStore
from escal8r.ticketlog import TicketLog
from escal8r.tracing import Tracer, load_spans

TICKETS = [
    {"id": "T005", "customer_plan": "Basis", "subject": "Nieuwe medewerker",
     "body": "Hoe voeg ik een nieuwe medewerker toe?"},
    {"id": "T003", "customer_plan": "Plus", "subject": "Dubbel afgeschreven",
     "body": "Twee keer betaald met kaart 4111 1111 1111 1111. Graag geld terug."},
    {"id": "T012", "customer_plan": "Plus", "subject": "Re:", "body": ""},
]  # fmt: skip


def fake_helpdesk() -> httpx.MockTransport:
    flaky = {"T003": 1}  # first POST for T003 gets a 503

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"tickets": TICKETS, "next_cursor": None})
        ticket = request.url.path.split("/")[-2]
        if flaky.get(ticket):
            flaky[ticket] -= 1
            return httpx.Response(503, json={"error": "service_unavailable"})
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handle)


def fake_openrouter() -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/chat/completions"):
            system, user = (m["content"] for m in body["messages"])
            if "extract evidence" in system:
                text = re.search(r"body: (.*)\n</ticket>", user, re.S).group(1)
                content = {
                    "language_guess": "nl",
                    "english_summary": "summary",
                    "requests": [{"text_en": "request", "quote": text[:20]}],
                    "faq_ids": ["F02"],
                }
            elif "draft_reply" in system:
                content = {"draft_reply": "Hoi, ga naar Team.", "reason": "F02 answer."}
            else:
                content = {"handoff_note": "Refund request.", "reason": "Client rule."}
            return httpx.Response(
                200,
                json={
                    "id": "gen-chat-1",
                    "model": body["model"],
                    "choices": [{"message": {"content": json.dumps(content)}}],
                    "usage": {"prompt_tokens": 900, "completion_tokens": 120, "cost": 0.0001},
                },
            )
        if set(body["questions"]) == set(VERIFY_QUESTIONS):
            answers = {
                "grounded": {"type": "noul", "noul": 0.95},
                "reply_language": {"type": "noul", "noul": 0.97},
                "commitment": {"type": "noul", "noul": 0.02},
            }
        else:
            refund = "terug" in body["state"]["ticket"]["body"]
            answers = jev_answers(refund_or_disputed_charge=0.9 if refund else 0.02)
        return httpx.Response(
            200,
            json={
                "id": "gen-dec-1",
                "model": body["model"] + "-20260917",
                "answers": answers,
                "usage": {"input_tokens": 2000, "output_tokens": 50, "cost": 0.00008},
            },
        )

    return httpx.MockTransport(handle)


@pytest.fixture
def traced_run(tmp_path):
    tracer = Tracer(tmp_path / "traces", "testrun")
    ticket_log = TicketLog(tmp_path / "logs", run_id="testrun")
    tracker = CostTracker(limit_usd=1.0)
    faq = load_faq(FAQ_PATH)
    with (
        HelpdeskClient(
            "https://helpdesk.test",
            "key",
            "me@example.com",
            transport=fake_helpdesk(),
            limiter=RateLimiter(1000, sleep=lambda _s: None),
            sleep=lambda _s: None,
        ) as desk,
        OpenRouterClient("sk", tracker, transport=fake_openrouter()) as models,
    ):
        # Only the run owner picks the tracer; clients and agent nest under its spans.
        agent = Agent(DeepSeek(models, faq), Jev(models), faq, tracker, Thresholds())
        summary = run(
            desk,
            StateStore(tmp_path / "state.json"),
            agent.decide,
            ticket_log=ticket_log,
            tracer=tracer,
        )
    log_lines = [json.loads(line) for line in ticket_log.path.read_text().splitlines()]
    return summary, load_trail(tracer.path), log_lines, tracer.path


def test_full_run_scores_100_percent(traced_run):
    summary, trail, log_lines, _ = traced_run
    assert summary.posted == ["T005", "T003", "T012"]
    report = check_run(trail, log_lines)
    assert report["failed"] == 0, format_check(report)
    assert report["tickets"] == 3
    assert report["score"] == 1.0
    # Every check really asserted something; nothing failed, so there was nothing to explain.
    exercised = {name for name, result in report["checks"].items() if result["passed"]}
    assert exercised == set(report["checks"]) - {"failures_explained"}


def test_trail_shows_every_step_of_a_reply(traced_run):
    _, trail, _, _ = traced_run
    text = render_trail(trail, "T005")
    for stage in ("decide", "redact", "interpret", "jev_decide", "rules", "write_reply"):
        assert f" {stage} " in text
    assert "jev_verify" in text and "deepseek_reply" in text
    assert "gen-dec-1" in text  # generation id, to look the call up in OpenRouter
    assert "POST /tickets/T005/reply" in text
    assert "rule=faq_reply" in text


def test_trail_shows_retry_and_redacted_text(traced_run):
    _, trail, _, _ = traced_run
    text = render_trail(trail, "T003", show_text=True)
    assert "503 (retry" in text  # the helpdesk retry is visible
    assert "rule=always_human" in text
    assert "[CARD]" in text
    assert "4111" not in text


def test_trace_file_never_holds_payment_details(traced_run):
    *_, path = traced_run
    assert "4111" not in path.read_text()


def test_log_line_points_at_its_trail(traced_run):
    _, trail, log_lines, _ = traced_run
    roots = {r["ticket"]: r["span"] for r in trail.roots()}
    assert {line["ticket"]: line["span"] for line in log_lines} == roots


# -- the eval catches broken trails ---------------------------------------------------


def _broken(trail: Trail, mutate) -> Trail:
    spans = json.loads(json.dumps(trail.spans))
    mutate(spans)
    return Trail(spans)


def _failed_checks(report: dict) -> set[str]:
    return {name for name, result in report["checks"].items() if result["failed"]}


def test_missing_stage_is_caught(traced_run):
    _, trail, log_lines, _ = traced_run
    broken = _broken(
        trail,
        lambda spans: spans.remove(
            next(s for s in spans if s["kind"] == "jev_verify" and s["ticket"] == "T005")
        ),
    )
    assert "stages_complete" in _failed_checks(check_run(broken, log_lines))


def test_cost_mismatch_is_caught(traced_run):
    _, trail, log_lines, _ = traced_run
    log_lines = json.loads(json.dumps(log_lines))
    log_lines[0]["trace"]["cost_usd"] += 0.01
    assert _failed_checks(check_run(trail, log_lines)) == {"cost_reconciles"}


def test_orphan_span_is_caught(traced_run):
    _, trail, log_lines, _ = traced_run
    broken = _broken(trail, lambda spans: spans[0].update(parent="nonexistent"))
    assert "tree_intact" in _failed_checks(check_run(broken, log_lines))


def test_leak_in_log_or_trail_is_caught(traced_run):
    _, trail, log_lines, _ = traced_run
    leaky_log = json.loads(json.dumps(log_lines))
    leaky_log[0]["draft_reply"] = "Hoi"
    broken = _broken(
        trail, lambda spans: spans[0].setdefault("text", {}).update(x="4111 1111 1111 1111")
    )
    assert "no_leaks" in _failed_checks(check_run(trail, leaky_log))
    assert "no_leaks" in _failed_checks(check_run(broken, log_lines))


def test_missing_log_line_is_caught(traced_run):
    _, trail, log_lines, _ = traced_run
    assert "log_matches_trail" in _failed_checks(check_run(trail, log_lines[1:]))


def test_unexplained_failure_is_caught(traced_run):
    _, trail, log_lines, _ = traced_run

    def mutate(spans):
        root = next(s for s in spans if s["kind"] == "ticket" and s["ticket"] == "T005")
        root["attrs"]["outcome"] = "post_failed"

    assert "failures_explained" in _failed_checks(check_run(_broken(trail, mutate), log_lines))


def test_model_call_without_cost_is_caught(traced_run):
    _, trail, log_lines, _ = traced_run

    def mutate(spans):
        next(s for s in spans if s["kind"] == "model_call")["attrs"].pop("cost_usd")

    assert "model_calls_complete" in _failed_checks(check_run(_broken(trail, mutate), None))


# -- tracer unit behaviour --------------------------------------------------------------


def test_spans_nest_and_inherit_ticket(tmp_path):
    tracer = Tracer(tmp_path, "r")
    with tracer.span("ticket", ticket_id="T1") as root, tracer.span("stage") as child:
        tracer.text(child, "prompt", "kaart 4111 1111 1111 1111")
    spans = {s["kind"]: s for s in load_spans(tracer.path)}
    assert spans["stage"]["parent"] == root.span_id
    assert spans["stage"]["ticket"] == "T1"
    assert spans["stage"]["text"]["prompt"] == "kaart [CARD]"


def test_exceptions_mark_the_span_failed(tmp_path):
    tracer = Tracer(tmp_path, "r")
    with pytest.raises(ValueError), tracer.span("stage"):
        raise ValueError("boom")
    [span] = load_spans(tracer.path)
    assert span["status"] == "error" and "boom" in span["error"]


def test_text_capture_can_be_turned_off(tmp_path):
    tracer = Tracer(tmp_path, "r", capture_text=False)
    with tracer.span("stage") as span:
        tracer.text(span, "prompt", "secret")
    assert "text" not in load_spans(tracer.path)[0]


def test_null_tracer_writes_nothing(tmp_path):
    from escal8r.tracing import NULL_TRACER

    with NULL_TRACER.span("stage") as span:
        span.set(a=1)
    assert NULL_TRACER.path is None


# -- CLI (local files only, no keys, no network) ----------------------------------------


def test_cli_trace_and_trace_check(traced_run, tmp_path, monkeypatch, capsys):
    from escal8r import cli

    *_, path = traced_run
    monkeypatch.setattr(cli, "TRACE_DIR", path.parent)
    monkeypatch.setattr(cli, "LOG_DIR", path.parent.parent / "logs")
    monkeypatch.setattr(cli, "EVAL_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    for name in ("API_KEY", "API_URL", "OPENROUTER_KEY"):
        monkeypatch.delenv(name, raising=False)

    assert cli.main(["trace", "T003", "--text"]) == 0
    out = capsys.readouterr().out
    assert "rule=always_human" in out and "[CARD]" in out

    assert cli.main(["trace-check"]) == 0
    out = capsys.readouterr().out
    assert "score 100.0%" in out
    assert (tmp_path / "results" / "observability-testrun.json").exists()

    assert cli.main(["trace", "T999"]) == 1


def test_calls_follow_the_current_span_not_a_bound_tracer(tmp_path):
    """Regression: two runs, two tracers, one set of clients -> no cross-run orphans."""
    from escal8r import tracing

    first, second = Tracer(tmp_path, "run1"), Tracer(tmp_path, "run2")
    for tracer in (first, second):
        with tracer.span("ticket", ticket_id="T1"), tracing.span("helpdesk"):
            pass
    for path in (first.path, second.path):
        report = check_run(load_trail(path), None)
        assert report["checks"]["tree_intact"]["failed"] == 0


def test_span_outside_a_run_records_nothing(tmp_path):
    from escal8r import tracing

    with tracing.span("helpdesk") as orphan:
        tracing.record_text(orphan, "request", "x")
    assert not list(tmp_path.iterdir())


def test_compare_two_runs(traced_run, tmp_path, monkeypatch, capsys):
    from escal8r import cli
    from escal8r.compare import run_metrics, ticket_diff

    _, trail, _, path = traced_run
    metrics = run_metrics(trail)
    assert (metrics["tickets"], metrics["reply"], metrics["handoff"], metrics["close"]) == (
        3,
        1,
        1,
        1,
    )
    assert metrics["verify_passed"] == 1 and metrics["posts"] == 3
    assert metrics["helpdesk_retries"] == 1  # the scripted 503 on T003
    assert metrics["trace_check_score"] == 1.0
    assert ticket_diff(trail, trail) == []

    monkeypatch.setattr(cli, "TRACE_DIR", path.parent)
    monkeypatch.setattr(cli, "EVAL_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    assert cli.main(["compare", "--a", "testrun", "--b", "testrun"]) == 0
    out = capsys.readouterr().out
    assert "| handoff_rate |" in out and "Tickets whose decision changed: 0" in out
