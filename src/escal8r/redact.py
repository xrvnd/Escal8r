"""Mask payment details before any model sees them and before anything is logged or posted.

Pattern matching alone would also mask order and phone numbers, so candidates are
validated: card numbers with the Luhn checksum, IBANs with the ISO 13616 mod-97 check.
"""

from __future__ import annotations

import re

CARD_TOKEN = "[CARD]"
IBAN_TOKEN = "[IBAN]"
CVC_TOKEN = "[CVC]"

# 13-19 digits, optionally grouped with single spaces or dashes.
_CARD_CANDIDATE = re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])")
# Country code, check digits, then 11-30 alphanumerics, optionally in groups of four.
_IBAN_CANDIDATE = re.compile(
    r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b", re.IGNORECASE
)
_CVC = re.compile(r"\b(CVC2?|CVV2?|CID)\b(\W{0,3})\d{3,4}\b", re.IGNORECASE)


def luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iban_valid(candidate: str) -> bool:
    compact = re.sub(r"\s", "", candidate).upper()
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(char, 36)) for char in rearranged)
    return int(numeric) % 97 == 1


def _mask_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    return CARD_TOKEN if 13 <= len(digits) <= 19 and luhn_valid(digits) else match.group()


def _mask_iban(match: re.Match[str]) -> str:
    return IBAN_TOKEN if iban_valid(match.group()) else match.group()


def redact(text: str) -> str:
    """Return `text` with valid card numbers, IBANs and CVC codes masked."""
    if not text:
        return text
    text = _CARD_CANDIDATE.sub(_mask_card, text)
    text = _IBAN_CANDIDATE.sub(_mask_iban, text)
    return _CVC.sub(lambda m: f"{m.group(1)}{m.group(2)}{CVC_TOKEN}", text)


def contains_payment_details(text: str) -> bool:
    """True if `text` holds an unmasked card number, IBAN or CVC."""
    return redact(text) != text
