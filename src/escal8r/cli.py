"""Command-line entry point: `uv run escal8r <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from escal8r.config import ConfigError, load_settings
from escal8r.helpdesk import HelpdeskClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="escal8r", description="Roosterly support ticket agent")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show helpdesk mode and open/handled counts")
    return parser


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
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    with HelpdeskClient.from_settings(settings) as client:
        if args.command == "status":
            print(json.dumps(client.status(), indent=2, ensure_ascii=False))
    return 0
