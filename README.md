# Escal8r

Reads every open Roosterly helpdesk ticket and posts one decision: **reply** (only from `faq.md`,
in the customer's language), **handoff** (English note, tagged with a team) or **close** (spam, empty).

## How to run

```bash
cp .env.example .env                 # API_KEY, API_URL, CANDIDATE_EMAIL, OPENROUTER_KEY
uv sync && uv run pytest             # offline tests
uv run escal8r eval --name final     # GET /examples only, never posts
uv run escal8r run --dry-run --all   # decide every ticket, post nothing
uv run escal8r reset --yes && uv run escal8r run
uv run escal8r trace T008 --text     # what happened to one ticket (local, redacted)
uv run escal8r trace-check           # is every ticket's trail complete?
```

## Decisions

1. **Two models, strict roles.** DeepSeek V4 Flash (`deepseek-v4-flash-0731`) only reads (requests
   with verified quotes, cited FAQ ids) and writes (temperature 0). Jev (`jev-1.14`) decides, in
   one typed call on the redacted ticket: language, category, a **route** over 14 buckets (7 FAQ
   topics, 6 human teams, spam), priority facts and 10 yes/no gates. Plain code does the rest.
2. **Priority is computed.** Jev answers facts (users affected, blocked now, workaround, deadline,
   cosmetic, risk gates); code applies the brief's P1–P4 table.
3. **Action rules, in order:** always-human gate (refund, cancellation, GDPR, security, legal,
   threat to leave) ≥ 0.35 → handoff; close only if spam gate *and* spam route agree; language or
   category confidence < 0.70 → handoff; P1/P2 → handoff; the route is judged by its whole
   distribution, not its top confidence: human routes ≥ 0.35 together, or FAQ beats human by
   < 0.15 → handoff; answerable and covered ≥ 0.75 → reply. Fixed, asymmetric, not tuned.
4. **Every reply is checked.** Jev sees what the writer saw (FAQ sections, customer plan, requests)
   and scores the draft: grounded (≥ 0.80), right language (≥ 0.80), no promise beyond the FAQ
   (< 0.35); else handoff. Cards, IBANs and CVCs are masked before any model and on every output.
   Ticket text is data; injected commands are ignored and noted.

**Dutch** is handled directly. **Assumptions:** no body and no subject beyond "Re:" → close; a
subject alone ("Refund please") is decided normally; a model failure → handoff. **AI tools:**
built with Claude Code; I made and reviewed the design choices.

## Reliability

Global limiter at 3.5 calls/s. **429** waits `Retry-After`; **503** backs off with jitter;
**409** marks the ticket done; a POST with a lost response is never blindly re-sent. **Crash-safe:**
the decision is saved (atomic write) before posting; on restart, a ticket still open gets the
*saved* decision. A file lock allows one run at a time; `kill -9` mid-run against a fake helpdesk
gave 0 duplicates. A bad key or model id, or `MAX_RUN_COST_USD`, stops the run. Secrets only from
env. `logs/` has one line per ticket, no text; each links to a local, gitignored trail in `traces/`.

## Eval and fix

10 labelled examples from `GET /examples`; labels never reach a model. Two or more runs per version,
since hosted models vary. Reports in [eval/results/](eval/results/).

| Version | lang | cat | priority | action | all correct | unsafe | handoffs |
|---|---|---|---|---|---|---|---|
| Baseline | 100% | 100% | 80% | 50–60% | 4–5 | 0 | 70–80% |
| Fix A: FAQ redirects | 100% | 100% | 80% | 60–70% | 5–6 | 0 | 60–70% |
| Fix B: priority wording | 100% | 100% | 100% | 80% | 8 | 0 | 50% |
| **Final** | 100% | 100% | 100% | 80% | 8 | 0 | 50% |

**Fix A.** E05 (idea) and E10 (maintenance) were "not answerable", though F10 and F11 say where
they go. **Fix B.** The brief's priority lines overlap ("money is involved" vs "how-to"), so E07
(price) and E01 (one device) got the wrong priority. **Final** keeps the score with priority computed
from facts, team-tagged routes and a plan-aware verifier. **Still missed**, as safe handoffs: E10
and E01 drafts under the 0.80 grounding bar; E01's human-route mass straddles 0.35.
**Final run:** 9 replies / 19 handoffs / 2 closes, all 30 posted, one 503 retried, trail check 100%,
both prompt injections (T007, T030) ignored.
**Cost per 100 tickets:** the final run's 101 model calls (137,468 tokens in, 23,072 out) cost
$0.00795 as reported by OpenRouter: $0.00795 / 30 × 100 = **$0.027** (other runs $0.026–0.028).
All runs together ≈ $0.08.

## What I would improve

Every miss is a safe-side handoff: 0 unsafe in all runs, but 50% handoffs vs 30% in the labels and
19 of 30 in the final run. Next: reply more often without lowering any bar. Reading the final run's
drafts and notes (`escal8r trace <id> --text`) shows where:

1. **The verifier rejects correct drafts.** T022 ("yes, German, under Profiel → Taal") matches F08
   but scored grounded 0.77; T016 (F10 roadmap) 0.79 vs 0.80. Jev gives a score, not a reason: ask
   one yes/no per sentence, rewrite only the failing one, verify again at 0.80 (also E10, E05).
2. **A rejected draft becomes a wrong note.** Told the draft "made unsupported claims", the note
   writer doubts the FAQ: T015's says no FAQ confirms Loket.nl (F04 does, on Plus), T022's asks a
   human to check German exists (F08), T016's suggests a delivery "timeline" that F10 rules out.
   Give the note writer the verified FAQ facts, and check notes the way replies are checked.
3. **Category doubt blocks a reply.** T020 (Shopify POS; route F10 at 0.96) went to a human only
   on category confidence 0.66 < 0.70. When every likely category gives the same action, skip it.
4. **German customers get Dutch menus.** T029 says tap "Ruil aanbieden" with no gloss (T022 had
   one). Always keep the FAQ's English gloss until the client gives German menu names.
5. **Readable, measured replies.** Log an English back-translation of every reply, and score the
   cited FAQ id in the eval, not only the four labels.
6. **Steadier runs.** OpenRouter picks the cheapest provider per call: pin one, and re-decide within
   ±0.05 of a line (E01: 0.27–0.41 vs 0.35). Then hand-label the 30 practice tickets and have a
   Dutch speaker check register (je/u).

**Brief change:** "P2: money is involved" also covers price questions the labels call P4 (E07);
I'd say "money is *disputed*".
