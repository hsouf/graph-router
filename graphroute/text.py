"""Tokenisation and lightweight term extraction.

No models, no network. Everything here is regex + string ops so that indexing
millions of files stays I/O bound rather than CPU bound.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Iterator, List

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")
# Ordered alternation, and the order matters:
#   OAuth-style two-cap prefix | acronym run (HTTPServer -> HTTP) | Word | word
_CAMEL_RE = re.compile(r"[A-Z][A-Z][a-z0-9]+|[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
_PROPER_RUN_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+)(?:\s+(?:[A-Z][a-z0-9]+)){1,3}\b")

# Small, deliberate stoplist. Anything genuinely corpus-specific gets removed by
# the document-frequency ceiling in the index builder instead.
STOPWORDS = frozenset(
    """
a about above after again against all also am an and any are as at be because been
before being below between both but by can cannot could did do does doing down during
each few for from further had has have having he her here hers herself him himself his
how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other others our ours ourselves out over own same she should so some such
than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves not none use used using via like via etc
""".split()
)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def split_identifier(token: str) -> List[str]:
    """`refreshOAuthToken` / `refresh_oauth_token` -> ['refresh','oauth','token'].

    Returns [] when the token has no internal structure worth splitting.
    """
    if "_" in token or "-" in token:
        parts = [p for p in re.split(r"[_\-]+", token) if p]
    else:
        parts = _CAMEL_RE.findall(token)
    if len(parts) < 2:
        return []
    return [p.lower() for p in parts if len(p) > 2]


def iter_terms(text: str, max_chars: int = 400_000) -> Iterator[str]:
    """Yield normalised terms for one document.

    Emits three families of term:
      * unigrams (lowercased, accent-stripped)
      * sub-tokens of compound identifiers (so `oauth_token` also matches `token`)
      * proper-noun runs joined by `_` (a cheap stand-in for named entities)
    """
    if len(text) > max_chars:
        text = text[:max_chars]

    for match in _PROPER_RUN_RE.finditer(text):
        phrase = "_".join(match.group(0).lower().split())
        yield phrase

    text = _strip_accents(text)
    for tok in _TOKEN_RE.findall(text):
        low = tok.lower()
        if len(low) < 3 or low in STOPWORDS:
            continue
        yield low
        for part in split_identifier(tok):
            if len(part) >= 3 and part not in STOPWORDS:
                yield part


def query_terms(query: str) -> List[str]:
    """Terms for a query string. Same pipeline, deduplicated, order preserved."""
    seen: dict = {}
    for t in iter_terms(query):
        seen.setdefault(t, None)
    return list(seen)


def count_terms(text: str) -> dict:
    counts: dict = {}
    for t in iter_terms(text):
        counts[t] = counts.get(t, 0) + 1
    return counts


def batch_count(texts: Iterable[str]) -> Iterator[dict]:
    for t in texts:
        yield count_terms(t)
