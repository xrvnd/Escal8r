#!/usr/bin/env python3
"""Pull all tickets (and labelled examples) from the Roosterly helpdesk API.

Read-only: only GET requests are made. Each ticket is saved as
tickets/<id>.json; examples go to tickets/examples/<id>.json.

Required env (from .env at the project root):
  API_KEY          bearer key
  API_URL          helpdesk base URL
  CANDIDATE_EMAIL  email you applied with (sent as X-Candidate-Email)
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tickets"
MAX_RETRIES = 6
MIN_INTERVAL = 0.25  # stay well under 5 req/s


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v[:1] in ('"', "'") and v[0] in v[1:]:
            v = v[1 : v.index(v[0], 1)]  # quoted: take text up to closing quote
        else:
            v = v.split(" #", 1)[0].strip()  # unquoted: drop inline comment
        os.environ.setdefault(k.strip(), v)


_last_call = 0.0


def get(base: str, path: str, headers: dict, params: dict | None = None) -> dict:
    global _last_call
    url = base.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(1, MAX_RETRIES + 1):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 429:
                delay = float(e.headers.get("Retry-After") or 1)
            elif e.code == 503 or e.code >= 500:
                delay = min(0.5 * 2 ** (attempt - 1), 8)
            else:
                sys.exit(f"GET {path} failed: HTTP {e.code} {body}")
            print(f"  {path}: HTTP {e.code}, retry {attempt}/{MAX_RETRIES} in {delay:.1f}s")
            time.sleep(delay)
        except urllib.error.URLError as e:
            delay = min(0.5 * 2 ** (attempt - 1), 8)
            print(f"  {path}: network error {e.reason}, retry in {delay:.1f}s")
            time.sleep(delay)
    sys.exit(f"GET {path} failed after {MAX_RETRIES} attempts")


def save(directory: Path, item: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{item['id']}.json").write_text(
        json.dumps(item, ensure_ascii=False, indent=2) + "\n"
    )


def main() -> None:
    load_env(ROOT / ".env")
    missing = [k for k in ("API_KEY", "API_URL", "CANDIDATE_EMAIL") if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)} (add them to .env)")

    base = os.environ["API_URL"]
    headers = {
        "Authorization": f"Bearer {os.environ['API_KEY']}",
        "X-Candidate-Email": os.environ["CANDIDATE_EMAIL"],
        "Accept": "application/json",
    }
    if os.environ.get("CANDIDATE_NAME"):
        headers["X-Candidate-Name"] = os.environ["CANDIDATE_NAME"]

    status = get(base, "/status", headers)
    print("status:", json.dumps(status))

    tickets, cursor = [], None
    while True:
        params = {"limit": 10, "status": "all"}
        if cursor:
            params["cursor"] = cursor
        page = get(base, "/tickets", headers, params)
        batch = page.get("tickets") or page.get("data") or page.get("items") or []
        tickets.extend(batch)
        print(f"  page: {len(batch)} tickets (total {len(tickets)})")
        cursor = page.get("next_cursor")
        if not cursor:
            break

    seen = set()
    for t in tickets:
        if t["id"] in seen:
            print(f"  duplicate id {t['id']} skipped")
            continue
        seen.add(t["id"])
        save(OUT_DIR, t)

    examples = get(base, "/examples", headers)
    ex_list = examples if isinstance(examples, list) else (
        examples.get("examples") or examples.get("data") or []
    )
    for ex in ex_list:
        save(OUT_DIR / "examples", ex)

    print(f"saved {len(seen)} tickets to {OUT_DIR}")
    print(f"saved {len(ex_list)} examples to {OUT_DIR / 'examples'}")


if __name__ == "__main__":
    main()
