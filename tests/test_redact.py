import pytest

from escal8r.redact import contains_payment_details, iban_valid, luhn_valid, redact


@pytest.mark.parametrize(
    "text",
    [
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
        "4111111111111111",
        "5500 0000 0000 0004",
    ],
)
def test_valid_cards_are_masked(text):
    assert redact(f"kaart {text} graag") == "kaart [CARD] graag"


@pytest.mark.parametrize(
    "text",
    [
        "order 1234567890123",  # fails Luhn
        "bel +31 6 12345678",  # too short
        "factuur 2026-10-0041",
        "4111 1111 1111 1112",  # one digit off
    ],
)
def test_lookalikes_are_kept(text):
    assert redact(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IBAN NL91 ABNA 0417 1643 00 graag", "IBAN [IBAN] graag"),
        ("NL91ABNA0417164300", "[IBAN]"),
        ("GB82 WEST 1234 5698 7654 32.", "[IBAN]."),
        ("NL91 ABNA 0417 1643 01", "NL91 ABNA 0417 1643 01"),  # bad checksum
    ],
)
def test_iban(text, expected):
    assert redact(text) == expected


def test_cvc_is_masked():
    assert redact("CVC 123, CVV: 4567") == "CVC [CVC], CVV: [CVC]"


def test_full_payment_ticket():
    body = "creditcard: 4111 1111 1111 1111, geldig tot 09/28, CVC 123."
    assert redact(body) == "creditcard: [CARD], geldig tot 09/28, CVC [CVC]."
    assert contains_payment_details(body)
    assert not contains_payment_details(redact(body))


def test_checksums():
    assert luhn_valid("4111111111111111") and not luhn_valid("4111111111111112")
    assert iban_valid("NL91ABNA0417164300") and not iban_valid("NL91ABNA0417164301")


def test_empty_text():
    assert redact("") == ""
