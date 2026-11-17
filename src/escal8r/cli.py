"""Command-line entry point: `uv run escal8r <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from escal8r.agent import Agent
from escal8r.config import (
    DEEPSEEK_MODEL,
    FAQ_PATH,
    JEV_MODEL,
    LOG_DIR,
    PROJECT_ROOT,
    STATE_DIR,
    ConfigError,
    Settings,
    load_settings,
)
from escal8r.deepseek import DeepSeek
from escal8r.evaluation import evaluate, format_report, save_report
from escal8r.faq import load_faq
from escal8r.helpdesk import HelpdeskClient
from escal8r.jev import Jev
from escal8r.observability import check_run, format_check, latest_trace, load_trail, render_trail
from escal8r.openrouter import CostTracker, OpenRouterClient, StopRun
from escal8r.runner import reset_all, run
from escal8r.state import StateError, StateStore
from escal8r.ticketlog import TicketLog
from escal8r.tracing import NULL_TRACER, Tracer

STATE_FILE = STATE_DIR / "state.json"
EVAL_DIR = PROJECT_ROOT / "eval" / "results"
TRACE_DIR = PROJECT_ROOT / "traces"

log = logging.getLogger("escal8r")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="escal8r", description="Roosterly support ticket agent")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="helpdesk counts and local state counts")

    run_cmd = commands.add_parser("run", help="handle every open ticket once")
    run_cmd.add_argument(
        "--dry-run", action="store_true", help="decide and log, but post and save nothing"
    )

    eval_cmd = commands.add_parser("eval", help="score the agent on GET /examples (no posting)")
    eval_cmd.add_argument("--name", default="latest", help="results file name in eval/results/")

    commands.add_parser("check-models", help="confirm the pinned model IDs exist on OpenRouter")

    trace = commands.add_parser("trace", help="show everything that happened to one ticket")
    trace.add_argument("ticket", help="ticket id, e.g. T008 or E03")
    trace.add_argument("--run", help="run id (default: the newest trail with this ticket)")
    trace.add_argument("--text", action="store_true", help="also show redacted prompts/outputs")

    check = commands.add_parser(
        "trace-check", help="observability eval: is every ticket's trail complete?"
    )
    check.add_argument("--run", help="run id (default: the newest trail)")

    reset = commands.add_parser(
        "reset", help="re-open all practice tickets and clear local state (destroys the last run)"
    )
    reset.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return parser


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _new_run_id(prefix: str = "") -> str:
    return prefix + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@contextmanager
def _agent(settings: Settings) -> Iterator[tuple[Agent, CostTracker]]:
    if not settings.openrouter_key:
        raise ConfigError("Missing environment variable: OPENROUTER_KEY. See .env.example.")
    tracker = CostTracker(limit_usd=settings.max_run_cost_usd)
    faq = load_faq(FAQ_PATH)
    with OpenRouterClient(settings.openrouter_key, tracker) as client:
        agent = Agent(DeepSeek(client, faq), Jev(client), faq, tracker, settings.thresholds)
        yield agent, tracker


def _cost_line(tracker: CostTracker, tickets: int) -> str:
    per_100 = 100 * tracker.spent_usd / tickets if tickets else 0.0
    estimated = any(not call.cost_reported for call in tracker.calls)
    return (
        f"model cost: ${tracker.spent_usd:.4f} for {len(tracker.calls)} calls, "
        f"${per_100:.4f} per 100 tickets" + (" (partly estimated)" if estimated else "")
    )


def _cmd_run(
    settings: Settings,
    helpdesk: HelpdeskClient,
    store: StateStore,
    tracer: Tracer,
    dry_run: bool,
) -> int:
    ticket_log = TicketLog(LOG_DIR, run_id=tracer.run_id)
    with _agent(settings) as (agent, tracker):
        summary = run(
            helpdesk, store, agent.decide, ticket_log=ticket_log, tracer=tracer, dry_run=dry_run
        )
    print(json.dumps({"counts": summary.counts(), "actions": summary.actions}, indent=2))
    if summary.stopped:
        print(f"RUN STOPPED: {summary.stopped}. {len(summary.not_started)} tickets left open.")
    print(_cost_line(tracker, sum(summary.actions.values())))
    print(f"log: {ticket_log.path.relative_to(PROJECT_ROOT)}")
    print(f"trail: traces/{tracer.run_id}.jsonl  (escal8r trace <ticket> / trace-check)")
    problems = summary.failed or summary.outcome_unknown or summary.not_started
    return 1 if problems else 0


def _cmd_eval(settings: Settings, helpdesk: HelpdeskClient, tracer: Tracer, name: str) -> int:
    examples = helpdesk.examples()
    with _agent(settings) as (agent, _tracker):
        report = evaluate(examples, agent.decide, tracer)
    path = save_report(report, EVAL_DIR, name)
    print(format_report(report))
    print(f"\nsaved: {path.relative_to(PROJECT_ROOT)}")
    print(f"trail: traces/{tracer.run_id}.jsonl")
    return 0


def _trace_file(run_id: str | None, ticket_id: str | None = None) -> Path | None:
    if run_id:
        path = TRACE_DIR / f"{run_id}.jsonl"
        return path if path.exists() else None
    return latest_trace(TRACE_DIR, ticket_id) if TRACE_DIR.exists() else None


def _cmd_trace(ticket_id: str, run_id: str | None, show_text: bool) -> int:
    path = _trace_file(run_id, ticket_id)
    if path is None:
        print(f"no trail found for {ticket_id}", file=sys.stderr)
        return 1
    print(render_trail(load_trail(path), ticket_id, show_text=show_text))
    return 0


def _cmd_trace_check(run_id: str | None) -> int:
    path = _trace_file(run_id)
    if path is None:
        print("no trail found; run `escal8r run` or `escal8r eval` first", file=sys.stderr)
        return 1
    log_path = LOG_DIR / f"run-{path.stem}.jsonl"
    log_lines = (
        [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
        if log_path.exists()
        else None  # eval runs have no ticket log
    )
    report = check_run(load_trail(path), log_lines)
    saved = save_report(report, EVAL_DIR, f"observability-{path.stem}")
    print(format_check(report))
    print(f"\nsaved: {saved.relative_to(PROJECT_ROOT)}")
    return 0 if report["failed"] == 0 else 1


def _cmd_check_models(settings: Settings) -> int:
    if not settings.openrouter_key:
        raise ConfigError("Missing environment variable: OPENROUTER_KEY. See .env.example.")
    with OpenRouterClient(settings.openrouter_key, CostTracker(limit_usd=1.0)) as client:
        listed = {model.get("id"): model for model in client.list_models()}
    ok = True
    for model_id in (JEV_MODEL, DEEPSEEK_MODEL):
        model = listed.get(model_id)
        if model is None:
            print(f"NOT LISTED  {model_id}")
            ok = False
        else:
            print(f"ok          {model_id}  pricing={json.dumps(model.get('pricing', {}))}")
    if not ok:
        print(
            "A pinned model is missing from /v1/models. Check openrouter.ai/models before running."
        )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # HTTP libraries log URLs and connection details; keep them quiet even with -v.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Local-only commands: read trails, need no keys, make no network calls.
    if args.command == "trace":
        return _cmd_trace(args.ticket, args.run, args.text)
    if args.command == "trace-check":
        return _cmd_trace_check(args.run)

    tracer = NULL_TRACER
    if args.command == "run":
        tracer = Tracer(TRACE_DIR, _new_run_id())
    elif args.command == "eval":
        tracer = Tracer(TRACE_DIR, _new_run_id(f"eval-{args.name}-"))

    try:
        settings = load_settings()
        if args.command == "check-models":
            return _cmd_check_models(settings)
        store = StateStore(STATE_FILE)
        with HelpdeskClient.from_settings(settings) as helpdesk:
            if args.command == "status":
                report = {"helpdesk": helpdesk.status(), "local_state": store.counts()}
                print(json.dumps(report, indent=2, ensure_ascii=False))
                return 0
            if args.command == "run":
                return _cmd_run(settings, helpdesk, store, tracer, args.dry_run)
            if args.command == "eval":
                return _cmd_eval(settings, helpdesk, tracer, args.name)
            if args.command == "reset":
                if not args.yes and not _confirm(
                    "Re-open all practice tickets and clear local state? This erases the last run."
                ):
                    print("aborted")
                    return 1
                reset_all(helpdesk, store)
                print("practice tickets re-opened, local state cleared")
                return 0
    except (ConfigError, StateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except StopRun as exc:  # eval or check-models hit a setup error or the budget
        print(f"stopped: {exc}", file=sys.stderr)
        return 3
    return 0
