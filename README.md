# Escal8r

Reads every open Roosterly helpdesk ticket and posts one decision: **reply** (FAQ-only draft in the
customer's language), **handoff** (English note) or **close** (spam, empty).

## How to run

```bash
cp .env.example .env              # API_KEY, CANDIDATE_EMAIL, OPENROUTER_KEY
uv sync && uv run pytest          # offline tests
uv run escal8r check-models       # pinned model IDs exist
uv run escal8r eval --name baseline   # GET /examples, never posts
uv run escal8r reset --yes        # re-open practice tickets + clear local state
uv run escal8r run                # add --dry-run to post nothing
uv run escal8r trace T008 --text  # everything that happened to one ticket (local, no keys)
uv run escal8r trace-check        # observability eval: is every ticket's trail complete?
```

## Decisions

1. **Two models, strict roles.** DeepSeek V4 Flash (`deepseek-v4-flash-20260731`) only reads
   (ticket → requests with quotes, FAQ ids, English summary) and writes. Jev (`jev-1.14`) makes
   every decision in one typed call: language, category, priority, 10 yes/no gates. Plain code
   picks the action. Jev reads the redacted ticket itself, so an interpreter miss can't hide a
   refund or GDPR request.
2. **Fixed, asymmetric thresholds, in order:** always-human gate (refund, cancellation, GDPR,
   security, legal, threat to leave) ≥ 0.35 → handoff; spam ≥ 0.85 → close; any confidence < 0.70
   → handoff; P1/P2 → handoff (policy); answerable and covered ≥ 0.75 → reply. Missing a
   human-only ticket costs most, so those gates trip low. Not tuned: 10 examples would overfit.
3. **Safety before posting.** Cards (Luhn), IBANs (mod-97) and CVCs are masked before any model
   call and on every output. Ticket text is untrusted data in every prompt; injected commands are
   ignored and noted in `reason`.

**Dutch.** Worked in directly, no translate step. The reply writer sees only verified requests and
cited FAQ sections. Jev then checks the draft: claims supported by those sections (≥ 0.80), right
language (≥ 0.80), no promise beyond the FAQ (< 0.35). Any failure → handoff. Interpreter quotes
must appear in the ticket, which catches invented evidence.
**AI tools:** built with Claude Code (planning, code, tests, offline rehearsal); I made and
reviewed the design choices. **Assumptions:** empty body → close as nl/other/P4; asking *how* to
upgrade is a how-to, not a contract change; if Jev fails → handoff as other/P3.

## Reliability

Key and email on every helpdesk call; one global limiter at 3.5 calls/s. **429** waits
`Retry-After`; **503** backs off exponentially with jitter (nothing was saved); **409** marks the
ticket done, never retried; a POST with a lost response is not retried. **Crash:** the decision is
saved `in_progress` (atomic write) before posting, `done` after. On restart, `in_progress` but no
longer open → done; still open → the same decision is re-sent. A file lock allows one run at a
time. Tested with `kill -9` mid-run against a local fake helpdesk: 30 handled, 0 duplicates.
A failing model call → templated handoff for that ticket; a setup error (400/401/402/403/404:
key, credits, model id, rejected parameter) or `MAX_RUN_COST_USD` stops the run, tickets left open. **Security:** secrets only from env (`.env` gitignored);
`provider.data_collection: "deny"` on OpenRouter (no training/storage; not zero-retention); logs
hold ids, decisions, scores and cost, never ticket or reply text. **Observability:** each log
line points to a local, gitignored trail (`traces/`): every stage, model call (served model,
generation id, tokens, cost, retries, redacted prompt/output) and helpdesk attempt. `trace-check`
asserts every trail is complete, reconciles costs with the log and finds no leaks.

## Eval and fix

10 labelled examples from `GET /examples`; labels never reach the models. Full output in
[eval/results/](eval/results/).

| Run | lang | category | priority | action | unsafe replies | handoff rate |
|---|---|---|---|---|---|---|
| Baseline | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| After fix | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

**Fix:** _chosen after the baseline._
**Cost per 100 tickets** (list prices; replace with measured): interpret 100 × (1.8k in × $0.0474/M
+ 0.3k out × $0.45/M) = $0.022 · Jev 100 × 2.2k × $0.042/M = $0.009 · write 95 × (0.8k in + 0.22k
out) = $0.013 · verify 40 × 0.9k = $0.002 → **≈ $0.046**.

## What I would improve

**Next:** answer the FAQ part of mixed tickets (FAQ question + bug) and hand off the rest; have a
Dutch speaker review a reply test set. **Brief change:** "P2: money is involved" also covers plain
price questions, which the labels call P4; I'd say "money is *disputed*".
