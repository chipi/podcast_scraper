"""Whether a query's language should change which legs of retrieval run (S2.9 / D-14).

THE DECISION, STATED UP FRONT: v1 does NOT drop the dense leg for a suspected non-English
query. It accepts dilution and records the signal instead. That is a choice, and this module
exists to hold the reasoning where the next person will find it.

WHY NOT SCRIPT DETECTION. D-14 already recorded that it cannot discriminate for tier 1 —
*inflación* and *inflation* are both Latin script, and every tier-1 language is. So the obvious
mechanism is unavailable before it is even evaluated.

WHY NOT A STOP-WORD HEURISTIC EITHER, YET. The asymmetry decides it:

- Wrongly dropping the dense leg on an ENGLISH query is a REGRESSION for the 678 English
  episodes that are the whole corpus today. Semantic search is the thing those users came for.
- Wrongly keeping it on a SPANISH query is DILUTION: the keyword leg still reaches the Spanish
  content through `segments_nonen`, and the dense leg adds some English results that rank
  alongside. Noise, not absence.

A short query is exactly where a stop-word heuristic is least reliable — "inflación" has no
stop words at all — and short queries are most of them. So a heuristic confident enough to be
worth its regression risk cannot be built from what a two-word query offers, and D-14 asked for
this to be "measured either way" while no query corpus exists to measure against.

WHAT THIS MODULE DOES PROVIDE: a cheap, honest signal that a query LOOKS non-English, recorded
so the decision can be revisited with data instead of argued again from first principles. When
enough queries carry the signal, the measurement D-14 wanted becomes possible.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Set

logger = logging.getLogger(__name__)

#: Characters that appear in tier-1 non-English text and essentially never in English.
#: Diacritics ARE weak evidence where script is none — but weak is the operative word: plenty of
#: Spanish is written without them, so their absence says nothing at all.
_NON_ENGLISH_CHARS = re.compile(r"[áàâãäéèêëíìîïóòôõöúùûüñçßøåæœ]", re.IGNORECASE)

#: Function words that are common in one tier-1 language and rare-to-absent in English. Only
#: useful on a query long enough to contain one, which most are not.
_MARKERS = {
    "es": {"de", "la", "el", "que", "los", "las", "una", "por", "para", "con", "del", "es"},
    "pt": {"de", "da", "do", "que", "uma", "para", "com", "não", "os", "as"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "mit", "für", "auf", "von"},
    "fr": {"le", "la", "les", "des", "une", "qui", "que", "pour", "avec", "pas"},
    "it": {"il", "la", "che", "di", "una", "per", "con", "non", "sono", "dei"},
}

#: English function words. A query carrying these is very unlikely to be non-English, and this
#: half matters more than the other: it is what stops an English query being misread.
_ENGLISH_MARKERS = {
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "it",
    "that",
    "for",
    "with",
    "was",
    "on",
    "are",
    "what",
    "how",
    "why",
    "who",
    "does",
    "did",
    "about",
}


@dataclass
class QueryLanguageSignal:
    """What the query looks like, and what retrieval should do about it.

    ``drop_dense_leg`` is ALWAYS False in v1. It exists as a field rather than being absent so
    the decision is visible at the call site and a future change is one line with a test behind
    it, rather than a new concept to introduce under time pressure.
    """

    looks_non_english: bool
    #: The language guessed, when a marker set matched. Never used to filter anything yet.
    guess: Optional[str] = None
    #: Why — for the log line and any later measurement.
    evidence: str = ""
    drop_dense_leg: bool = False


def _tokens(text: str) -> Set[str]:
    return {w for w in re.findall(r"[^\W\d_]+", (text or "").lower(), re.UNICODE) if w}


def assess_query_language(text: str) -> QueryLanguageSignal:
    """A cheap signal about a query's language. Never changes which legs run, in v1.

    Deliberately conservative in one direction: any English function word suppresses the
    signal entirely. A false "non-English" on an English query is the expensive mistake, and
    the cheap mistake is the one this tolerates.
    """
    if not text or not text.strip():
        return QueryLanguageSignal(looks_non_english=False, evidence="empty query")

    tokens = _tokens(text)
    if tokens & _ENGLISH_MARKERS:
        return QueryLanguageSignal(
            looks_non_english=False,
            evidence="carries an English function word",
        )

    # MOST HITS WINS, not first-in-dict. The marker sets overlap by design — "una" is Spanish
    # AND Italian, "de"/"que" are Spanish AND Portuguese — so iterating in declaration order
    # handed "il che una" to `es` on one shared token while `it` matched three. The guess is
    # recorded for the measurement D-14 asked for, so a systematically wrong guess would poison
    # exactly the data it exists to collect. Ties break on the language name, for determinism.
    scored = sorted(
        ((len(tokens & markers), lang) for lang, markers in _MARKERS.items() if tokens & markers),
        key=lambda pair: (-pair[0], pair[1]),
    )
    if scored:
        _count, language = scored[0]
        hits = tokens & _MARKERS[language]
        return QueryLanguageSignal(
            looks_non_english=True,
            guess=language,
            evidence=f"{language} function words: {sorted(hits)}",
        )

    if _NON_ENGLISH_CHARS.search(text):
        return QueryLanguageSignal(
            looks_non_english=True,
            evidence="non-English diacritics",
        )

    # The common case for a short query: genuinely ambiguous. Saying so is the honest answer,
    # and it is why the dense leg is not dropped.
    return QueryLanguageSignal(looks_non_english=False, evidence="no discriminating signal")
