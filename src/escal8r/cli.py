"""Command-line entry point: `uv run escal8r <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from escal8r.config import STATE_DIR, ConfigError, load_settings
from escal8r.helpdesk import HelpdeskClient
from escal8r.runner import reset_all
from escal8r.state import StateError, StateStore

STATE_FILE = STATE_DIR / "state.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="escal8r", description="Roosterly support ticket agent")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="helpdesk counts and local state counts")
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # HTTP libraries log URLs and connection details; keep them quiet even with -v.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        settings = load_settings()
        store = StateStore(STATE_FILE)
    except (ConfigError, StateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with HelpdeskClient.from_settings(settings) as client:
        if args.command == "status":
            report = {"helpdesk": client.status(), "local_state": store.counts()}
            print(json.dumps(report, indent=2, ensure_ascii=False))
        elif args.command == "reset":
            if not args.yes and not _confirm(
                "Re-open all practice tickets and clear local state? This erases the last run."
            ):
                print("aborted")
                return 1
            try:
                reset_all(client, store)
            except StateError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print("practice tickets re-opened, local state cleared")
    return 0
