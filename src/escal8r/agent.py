"""The per-ticket pipeline.

redact -> empty? close -> DeepSeek interpret -> Jev decide -> rules pick action
       -> DeepSeek write -> (reply only) Jev verify -> redact output -> Decision

Any model failure ends in a handoff, never in a guessed reply. The `trace` dict
records numbers and names only (no ticket or reply text) for the committed log.
Each stage is also a span in the local trail (see tracing.py).
"""

from __future__ import annotations

import re
from typing import Any, cast

from escal8r.config import LANGUAGE_NAMES, Thresholds
from escal8r.deepseek import DeepSeek, Interpretation
from escal8r.faq import Faq
from escal8r.jev import ROUTES, Jev, JevDecision
from escal8r.models import Category, Decided, Decision, Language, Priority, Ticket
from escal8r.openrouter import CostTracker, ModelError
from escal8r.redact import contains_payment_details, redact
from escal8r.rules import (
    DEFAULT_QUEUE,
    compute_priority,
    pick_action,
    reply_check_failure,
    reply_sections,
)
from escal8r.tracing import record_text, span

# Used only when Jev itself fails and gives us no classification.
FALLBACK_CATEGORY: Category = "other"
FALLBACK_PRIORITY: Priority = "P3"
DEFAULT_LANGUAGE: Language = "nl"  # most Roosterly customers write Dutch
INJECTION_NOTE = " The ticket contained instructions aimed at an AI; they were ignored."
INJECTION_THRESHOLD = 0.5

# Reply/forward prefixes (English, Dutch, German) and punctuation carry no request.
_REPLY_PREFIXES = re.compile(r"\b(re|fw|fwd|aw|wg|antw|doorst)\s*:", re.IGNORECASE)


def _is_empty(ticket: Ticket) -> bool:
    """Nothing to act on: no body, and a subject that is blank or only "Re:"-style prefixes.
    A subject like "Refund please" with no body is NOT empty: it goes through the pipeline."""
    subject = _REPLY_PREFIXES.sub("", ticket.subject)
    return not ticket.body.strip() and not any(char.isalnum() for char in subject)


def _fallback_language(interpretation: Interpretation | None) -> Language:
    guess = interpretation.language_guess if interpretation else ""
    return cast(Language, guess) if guess in LANGUAGE_NAMES else DEFAULT_LANGUAGE


class Agent:
    def __init__(
        self,
        deepseek: DeepSeek,
        jev: Jev,
        faq: Faq,
        tracker: CostTracker,
        thresholds: Thresholds,
    ):
        self._deepseek = deepseek
        self._jev = jev
        self._faq = faq
        self._tracker = tracker
        self._t = thresholds

    def decide(self, ticket: Ticket) -> Decided:
        first_call = len(self._tracker.calls)
        trace: dict[str, Any] = {}
        with span("decide", ticket_id=ticket.id) as stage:
            decision = self._decide(ticket, trace)
            stage.set(
                rule=trace.get("rule"),
                action=decision.action,
                language=decision.language,
                category=decision.category,
                priority=decision.priority,
                confidence=decision.confidence,
            )
            record_text(stage, "reason", decision.reason)
            record_text(stage, "draft_reply", decision.draft_reply)
            record_text(stage, "handoff_note", decision.handoff_note)
        calls = self._tracker.since(first_call)
        trace["models"] = sorted({call.model_served for call in calls})
        trace["model_calls"] = len(calls)
        trace["tokens"] = {
            "input": sum(call.input_tokens for call in calls),
            "output": sum(call.output_tokens for call in calls),
        }
        trace["cost_usd"] = round(sum(call.cost_usd for call in calls), 6)
        trace["cost_estimated"] = any(not call.cost_reported for call in calls)
        return Decided(decision=decision, trace=trace)

    # -- pipeline --------------------------------------------------------------

    def _decide(self, ticket: Ticket, trace: dict[str, Any]) -> Decision:
        with span("redact") as stage:
            safe = ticket.model_copy(
                update={"subject": redact(ticket.subject), "body": redact(ticket.body)}
            )
            trace["payment_details_redacted"] = safe != ticket
            stage.set(payment_details_redacted=safe != ticket, plan=safe.customer_plan)
            record_text(stage, "ticket", f"subject: {safe.subject}\nbody: {safe.body}")

        if _is_empty(safe):
            trace["rule"] = "empty"
            return Decision(
                action="close",
                language=DEFAULT_LANGUAGE,
                category="other",
                priority="P4",
                reason="Empty ticket: no message and no subject, so there is nothing to answer.",
                confidence=1.0,
            )

        interpretation = self._interpret(safe, trace)

        jev: JevDecision | None = None
        with span("jev_decide") as stage:
            try:
                decided = self._jev.decide(self._decision_state(safe, interpretation))
            except ModelError as exc:
                stage.fail(str(exc))
                trace["error"] = str(exc)
            else:
                stage.set(**decided.summary())
                jev = decided
        if jev is None:
            trace["rule"] = "decision_model_failed"
            return self._templated_handoff(
                safe,
                interpretation,
                _fallback_language(interpretation),
                FALLBACK_CATEGORY,
                FALLBACK_PRIORITY,
                why="the decision model failed",
                confidence=0.0,
            )

        trace["jev"] = jev.summary()
        language, category = jev.language_code, jev.category_code
        confidence = min(jev.min_confidence, jev.route.confidence)
        injection = jev.gates["injection_attempt"] >= INJECTION_THRESHOLD

        with span("rules") as stage:
            priority, priority_why = compute_priority(jev, self._t)
            outcome = pick_action(
                jev,
                self._t,
                has_requests=bool(interpretation and interpretation.requests),
                priority=priority,
            )
            stage.set(
                rule=outcome.rule,
                action=outcome.action,
                why=outcome.why,
                route=outcome.route,
                queue=outcome.queue,
                priority=priority,
                priority_why=priority_why,
            )
        trace["rule"] = outcome.rule
        trace["route"] = outcome.route
        trace["priority_why"] = priority_why

        if outcome.action == "close":
            return Decision(
                action="close",
                language=language,
                category=category,
                priority=priority,
                reason=f"{outcome.why} Spam probability {jev.gates['spam']:.2f}.",
                confidence=round(jev.gates["spam"], 3),
            )

        why = outcome.why
        if outcome.action == "reply" and interpretation is not None:
            reply = self._try_reply(interpretation, jev, priority, outcome.route, safe, trace)
            if isinstance(reply, Decision):
                if injection:
                    reply = reply.model_copy(update={"reason": reply.reason + INJECTION_NOTE})
                return reply
            trace["rule"] = "reply_rejected"
            trace["reply_rejected"] = reply
            why = f"A draft reply was rejected: {reply}."

        note_why = why + (INJECTION_NOTE if injection else "")
        queue = outcome.queue or ROUTES[outcome.route].queue or DEFAULT_QUEUE
        decision = self._handoff(
            safe, interpretation, language, category, priority, note_why, confidence, trace
        )
        # Transfer target first, so the support team sees where it belongs at a glance.
        note = f"[{queue}] {decision.handoff_note}"
        return decision.model_copy(update={"handoff_note": note})

    def _interpret(self, ticket: Ticket, trace: dict[str, Any]) -> Interpretation | None:
        with span("interpret") as stage:
            try:
                interpretation, dropped = self._deepseek.interpret(ticket)
            except ModelError as exc:
                stage.fail(str(exc))
                trace["interpret_error"] = str(exc)
                return None  # Jev still reads the ticket itself
            trace["requests_verified"] = len(interpretation.requests)
            trace["quotes_dropped"] = dropped
            trace["faq_ids"] = interpretation.faq_ids
            stage.set(
                requests_verified=len(interpretation.requests),
                quotes_dropped=dropped,
                faq_ids=interpretation.faq_ids,
                language_guess=interpretation.language_guess,
                embedded_instructions=len(interpretation.embedded_instructions),
            )
            return interpretation

    def _decision_state(
        self, ticket: Ticket, interpretation: Interpretation | None
    ) -> dict[str, Any]:
        return {
            "ticket": {
                "subject": ticket.subject,
                "body": ticket.body,
                "customer_plan": ticket.customer_plan,
                "channel": ticket.channel,
            },
            "interpreter_evidence": (
                interpretation.model_dump()
                if interpretation
                else "unavailable: read the ticket directly"
            ),
            "help_centre_faq": self._faq.render(),
            "support_team_rules": self._faq.rules,
        }

    def _try_reply(
        self,
        interpretation: Interpretation,
        jev: JevDecision,
        priority: Priority,
        route: str,
        ticket: Ticket,
        trace: dict[str, Any],
    ) -> Decision | str:
        """A verified reply, or the reason no reply may be sent."""
        sections = reply_sections(route, interpretation.faq_ids)
        if sections is None:
            return f"route {route} and the cited FAQ sections disagree"
        trace["reply_sections"] = sections
        language = jev.language_code
        with span("write_reply", faq_ids=sections) as stage:
            try:
                draft, reason = self._deepseek.write_reply(
                    language=language,
                    category=jev.category_code,
                    priority=priority,
                    customer_plan=ticket.customer_plan,
                    requests=interpretation.requests,
                    faq_ids=sections,
                )
            except ModelError as exc:
                stage.fail(str(exc))
                return f"the writer failed ({exc})"
            if contains_payment_details(draft):
                stage.fail("draft contained payment details")
                return "the draft contained payment details"

        with span("jev_verify") as stage:
            try:
                scores = self._jev.verify_reply(
                    {
                        "expected_language": LANGUAGE_NAMES[language],
                        "faq_sections": self._faq.render(sections),
                        # The writer saw the plan, so plan claims must be checkable too.
                        "customer_plan": ticket.customer_plan,
                        "customer_requests": [r.text_en for r in interpretation.requests],
                        "draft_reply": draft,
                    }
                )
            except ModelError as exc:
                stage.fail(str(exc))
                return f"verification failed ({exc})"
            trace["verify"] = {name: round(value, 3) for name, value in scores.items()}
            failure = reply_check_failure(scores, self._t)
            stage.set(scores=trace["verify"], passed=failure is None, failure=failure)
        if failure:
            return failure

        return Decision(
            action="reply",
            language=language,
            category=jev.category_code,
            priority=priority,
            reason=redact(reason),
            draft_reply=draft,
            confidence=round(min(jev.min_confidence, jev.route.confidence, scores["grounded"]), 3),
        )

    def _handoff(
        self,
        ticket: Ticket,
        interpretation: Interpretation | None,
        language: Language,
        category: Category,
        priority: Priority,
        why: str,
        confidence: float,
        trace: dict[str, Any],
    ) -> Decision:
        summary = interpretation.english_summary if interpretation else ""
        with span("write_handoff") as stage:
            try:
                note, reason = self._deepseek.write_handoff(
                    ticket, category=category, priority=priority, why=why, summary=summary
                )
            except ModelError as exc:
                stage.fail(str(exc))
                note = ""
        if not note:
            trace["handoff_writer_error"] = "handoff writer failed"
            return self._templated_handoff(
                ticket, interpretation, language, category, priority, why=why, confidence=confidence
            )
        return Decision(
            action="handoff",
            language=language,
            category=category,
            priority=priority,
            reason=redact(reason),
            handoff_note=redact(note),
            confidence=round(confidence, 3),
        )

    def _templated_handoff(
        self,
        ticket: Ticket,
        interpretation: Interpretation | None,
        language: Language,
        category: Category,
        priority: Priority,
        *,
        why: str,
        confidence: float,
    ) -> Decision:
        """Handoff without the writer. Internal text only; never sent to a customer."""
        summary = (
            interpretation.english_summary
            if interpretation and interpretation.english_summary
            else "not available"
        )
        note = (
            f"Automatic triage could not finish: {why.rstrip('.')}. "
            f"Summary: {summary} Subject: '{ticket.subject}'. Please read the ticket."
        )
        with span("templated_handoff", why=why):
            pass
        return Decision(
            action="handoff",
            language=language,
            category=category,
            priority=priority,
            reason=f"Handed to a human because {why.rstrip('.')}.",
            handoff_note=redact(note),
            confidence=round(confidence, 3),
        )
