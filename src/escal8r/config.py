"""Runtime configuration: secrets from the environment, everything else pinned here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAQ_PATH = PROJECT_ROOT / "knowledge" / "faq.md"
STATE_DIR = PROJECT_ROOT / "state"
LOG_DIR = PROJECT_ROOT / "logs"

# Pinned model IDs (OpenRouter). Never use floating aliases: they can be rerouted.
JEV_MODEL = "typesafe/jev-1.14"
DEEPSEEK_MODEL = "deepseek/deepseek-v4-flash-20260731"

# The helpdesk rejects more than 5 calls/s. At 4/s a sliding 1 s window can still hold
# 5 calls (t = 0, .25, .5, .75, 1.0); 3.5/s keeps every window at 4 or fewer.
HELPDESK_CALLS_PER_SECOND = 3.5


@dataclass(frozen=True)
class Thresholds:
    """Fixed decision thresholds. Asymmetric on purpose; see README > Decisions."""

    always_human: float = 0.35  # any always-human gate at/above this -> handoff
    spam: float = 0.85  # closing a real customer is costly, so spam must be clear
    choice_confidence: float = 0.70  # language/category/priority confidence floor
    answerable: float = 0.75  # FAQ answers the question
    covered: float = 0.75  # FAQ covers every request in the ticket
    grounded: float = 0.80  # every claim in the draft is supported by the FAQ


class ConfigError(RuntimeError):
    """Required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    helpdesk_url: str
    # Secrets and personal data are excluded from repr so they never reach logs.
    helpdesk_key: str = field(repr=False)
    candidate_email: str = field(repr=False)
    candidate_name: str | None = field(default=None, repr=False)
    openrouter_key: str | None = field(default=None, repr=False)
    max_run_cost_usd: float = 0.10
    thresholds: Thresholds = field(default_factory=Thresholds)


_REQUIRED = ("API_URL", "API_KEY", "CANDIDATE_EMAIL")


def load_settings(env_file: Path | None = PROJECT_ROOT / ".env") -> Settings:
    """Load settings from the environment, after reading `.env` if present.

    Real environment variables take precedence over `.env`.
    """
    if env_file is not None and env_file.exists():
        load_dotenv(env_file, override=False)

    missing = [name for name in _REQUIRED if not os.environ.get(name, "").strip()]
    if missing:
        raise ConfigError(f"Missing environment variables: {', '.join(missing)}. See .env.example.")

    try:
        max_cost = float(os.environ.get("MAX_RUN_COST_USD", "0.10"))
    except ValueError as exc:
        raise ConfigError("MAX_RUN_COST_USD must be a number") from exc

    return Settings(
        helpdesk_url=os.environ["API_URL"].strip().rstrip("/"),
        helpdesk_key=os.environ["API_KEY"].strip(),
        candidate_email=os.environ["CANDIDATE_EMAIL"].strip(),
        candidate_name=os.environ.get("CANDIDATE_NAME", "").strip() or None,
        openrouter_key=os.environ.get("OPENROUTER_KEY", "").strip() or None,
        max_run_cost_usd=max_cost,
    )
