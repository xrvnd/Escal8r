"""Parse the help centre (knowledge/faq.md) into numbered sections plus the support-team rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SECTION = re.compile(r"^## (F\d{2}) · (.+)$", re.MULTILINE)
_RULES_HEADING = "## The client's rules for the support team"


@dataclass(frozen=True)
class FaqSection:
    id: str
    title: str
    text: str

    def render(self) -> str:
        return f"{self.id} · {self.title}\n{self.text}"


@dataclass(frozen=True)
class Faq:
    sections: dict[str, FaqSection]
    rules: str

    def render(self, ids: list[str] | None = None) -> str:
        """The chosen sections (all by default) as plain text."""
        chosen = self.sections.values() if ids is None else [self.sections[i] for i in ids]
        return "\n\n".join(section.render() for section in chosen)

    def known(self, ids: list[str]) -> list[str]:
        """`ids` filtered to sections that exist, de-duplicated, order kept."""
        return [i for i in dict.fromkeys(ids) if i in self.sections]


def parse_faq(markdown: str) -> Faq:
    body, _, rules = markdown.partition(_RULES_HEADING)
    if not rules.strip():
        raise ValueError("FAQ has no support-team rules section")

    matches = list(_SECTION.finditer(body))
    if not matches:
        raise ValueError("FAQ has no '## Fxx · Title' sections")

    sections: dict[str, FaqSection] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[match.end() : end].strip().removesuffix("---").strip()
        sections[match.group(1)] = FaqSection(match.group(1), match.group(2).strip(), text)
    return Faq(sections=sections, rules=rules.strip())


def load_faq(path: Path) -> Faq:
    return parse_faq(path.read_text(encoding="utf-8"))
