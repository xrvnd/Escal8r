"""Typed shapes for helpdesk tickets, labelled examples and outgoing decisions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Language = Literal["nl", "en", "de"]
Category = Literal["bug", "billing", "feature_request", "question", "other"]
Priority = Literal["P1", "P2", "P3", "P4"]
Action = Literal["reply", "handoff", "close"]

# (has draft_reply, has handoff_note) required for each action.
_TEXT_FIELDS_BY_ACTION: dict[str, tuple[bool, bool]] = {
    "reply": (True, False),
    "handoff": (False, True),
    "close": (False, False),
}
_TEXT_RULE = {
    "reply": "a draft_reply and no handoff_note",
    "handoff": "a handoff_note and no draft_reply",
    "close": "no draft_reply and no handoff_note",
}


class Ticket(BaseModel):
    """A customer ticket. Its text is untrusted data, never instructions."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    received_at: str | None = None
    channel: str | None = None
    customer_plan: str | None = None
    subject: str = ""
    body: str = ""

    @field_validator("subject", "body", mode="before")
    @classmethod
    def _none_to_empty(cls, value: Any) -> Any:
        return "" if value is None else value


class Label(BaseModel):
    """The helpdesk's own label for an example ticket (eval ground truth)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    language: Language
    category: Category
    priority: Priority
    action: Action
    reason: str = ""


class Example(Ticket):
    label: Label


class Decision(BaseModel):
    """One decision for one ticket, validated before it is sent."""

    model_config = ConfigDict(frozen=True)

    action: Action
    language: Language
    category: Category
    priority: Priority
    reason: str = Field(min_length=1)
    draft_reply: str | None = None
    handoff_note: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _text_matches_action(self) -> Decision:
        has_reply = bool(self.draft_reply and self.draft_reply.strip())
        has_note = bool(self.handoff_note and self.handoff_note.strip())
        if (has_reply, has_note) != _TEXT_FIELDS_BY_ACTION[self.action]:
            raise ValueError(f"action '{self.action}' requires {_TEXT_RULE[self.action]}")
        return self

    def payload(self) -> dict[str, Any]:
        """Request body for POST /tickets/{id}/{action}."""
        return self.model_dump(exclude={"action"}, exclude_none=True)
