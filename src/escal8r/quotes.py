"""Check that evidence quoted by the interpreter really appears in the ticket.

A strict byte match would drop valid quotes over curly quotes, Unicode forms,
spacing or case, so both sides are normalised first.
"""

from __future__ import annotations

import re
import unicodedata

_QUOTE_CHARS = str.maketrans(
    {"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "‟": '"'}
)
_WHITESPACE = re.compile(r"\s+")
_EDGE_PUNCTUATION = " \"'.,;:!?…-"


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_CHARS)
    return _WHITESPACE.sub(" ", text).strip().casefold()


def quote_in_text(quote: str, text: str) -> bool:
    """True if `quote` (ignoring surrounding quote marks/ellipses) occurs in `text`."""
    needle = normalise(quote).strip(_EDGE_PUNCTUATION)
    return bool(needle) and needle in normalise(text)
