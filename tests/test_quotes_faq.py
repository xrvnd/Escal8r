import pytest

from escal8r.config import FAQ_PATH
from escal8r.faq import load_faq, parse_faq
from escal8r.quotes import quote_in_text

TICKET = "Hoi, ik ben mijn  wachtwoord vergeten en het manageraccount staat op mijn naam."


@pytest.mark.parametrize(
    "quote",
    [
        "ik ben mijn wachtwoord vergeten",  # extra space in ticket
        "Ik Ben Mijn Wachtwoord Vergeten",  # case
        "“mijn wachtwoord vergeten”",  # curly quotes around it
        "…het manageraccount staat op mijn naam.",  # ellipsis + punctuation
    ],
)
def test_quote_matches_after_normalising(quote):
    assert quote_in_text(quote, TICKET)


@pytest.mark.parametrize("quote", ["ik wil mijn geld terug", "", "   "])
def test_invented_or_empty_quote_fails(quote):
    assert not quote_in_text(quote, TICKET)


def test_curly_apostrophe_in_ticket():
    assert quote_in_text("de knop 'Ruil aanbieden'", "De knop ‘Ruil aanbieden’ doet niks")


def test_real_faq_parses():
    faq = load_faq(FAQ_PATH)
    assert list(faq.sections) == [f"F{i:02d}" for i in range(1, 13)]
    assert faq.sections["F05"].title == "Plans and prices"
    assert "€4 per active employee" in faq.sections["F05"].text
    assert "refunds or disputed charges" in faq.rules
    assert "---" not in faq.sections["F12"].text


def test_known_filters_and_dedupes():
    faq = load_faq(FAQ_PATH)
    assert faq.known(["F06", "F99", "F06", "F01"]) == ["F06", "F01"]


def test_render_subset():
    rendered = load_faq(FAQ_PATH).render(["F11"])
    assert rendered.startswith("F11 · Outages and maintenance")
    assert "F01" not in rendered


def test_faq_without_rules_is_rejected():
    with pytest.raises(ValueError):
        parse_faq("## F01 · Title\ntext")
