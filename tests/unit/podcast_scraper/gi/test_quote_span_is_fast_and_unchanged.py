"""Quote alignment returns the same span as before, without a regex scan per sub-phrase (#2207).

On prod (2026-09-30) one episode held the single pipeline slot ~34 minutes between two bundled
``extract_quotes`` calls, CPU-bound, no network. ``_subphrase_span`` compiled and ran one regex over
the whole transcript for every (start, end) sub-phrase of a quote that was not verbatim:
O(W² · len(transcript)). Measured on that episode's 67k-char transcript: 0.67 s for a 60-word
near-verbatim quote, 3.4 s at 120 words, 12 s at 200.

The token-index matcher must return EXACTLY what the regex matcher did — offsets feed timestamps,
evidence spans and dedupe — so the old one is kept as the oracle and compared on randomized text
built to hit every edge: suffix/prefix edge words, repeated phrases, overlapping repeats, case,
punctuation, odd whitespace.
"""

from __future__ import annotations

import random
import time

import pytest

from podcast_scraper.gi.grounding import (
    _subphrase_span,
    _subphrase_span_regex,
    resolve_llm_quote_span,
)

pytestmark = [pytest.mark.unit]

VOCAB = [
    "the", "The", "a", "of", "and", "And,", "war", "Rome", "Rome's", "general", "Caesar.",
    "said", "ha", "ha.", "un", "unless", "less", "it's", "it’s", "—", "(and)", "Sulla", "SULLA",
]  # fmt: skip
SPACES = [" ", " ", " ", "  ", "\n", "\t", " \n "]


def _text(rng: random.Random, n: int) -> str:
    out = []
    for _ in range(n):
        out.append(rng.choice(VOCAB))
        out.append(rng.choice(SPACES))
    return "".join(out).strip()


def _quote_from(rng: random.Random, transcript: str) -> str:
    """A slice of the transcript, then mangled the way a model restates a passage."""
    toks = transcript.split()
    a = rng.randrange(len(toks))
    b = min(len(toks), a + rng.randrange(2, 25))
    words = toks[a:b]
    out = []
    for w in words:
        roll = rng.random()
        if roll < 0.12:
            continue  # dropped word
        if roll < 0.22:
            out.append(rng.choice(VOCAB))  # substituted word
        elif roll < 0.27:
            out.append(w[1:] or w)  # clipped front (suffix-match edge)
        elif roll < 0.32:
            out.append(w[:-1] or w)  # clipped end (prefix-match edge)
        elif roll < 0.38:
            out.append(w.upper())
        else:
            out.append(w)
    return " ".join(out)


@pytest.mark.parametrize("seed", range(40))
def test_same_span_as_the_regex_matcher(seed: int) -> None:
    rng = random.Random(seed)
    transcript = _text(rng, rng.randrange(20, 400))
    for _ in range(60):
        qt = _quote_from(rng, transcript) if rng.random() < 0.8 else _text(rng, 12)
        for min_words in (2, 3):
            assert _subphrase_span(transcript, qt, min_words) == _subphrase_span_regex(
                transcript, qt, min_words
            ), (transcript, qt, min_words)


def test_overlapping_repeats_follow_finditer() -> None:
    """finditer skips a match overlapping the previous one; the index must skip it too."""
    transcript = "ha ha ha ha.  ha ha"
    for qt in ("ha ha ha", "ha ha", "x ha ha y"):
        assert _subphrase_span(transcript, qt, 2) == _subphrase_span_regex(transcript, qt, 2)


def test_a_length_changing_case_fold_uses_the_regex() -> None:
    transcript = "İstanbul is large and İstanbul is old"
    qt = "İstanbul is old"
    assert _subphrase_span(transcript, qt, 2) == _subphrase_span_regex(transcript, qt, 2)


def test_a_long_near_verbatim_quote_is_fast() -> None:
    """The shape that cost minutes: a long passage with one word changed every ten."""
    rng = random.Random(7)
    words = [f"w{rng.randrange(4000)}" for _ in range(12000)]  # ~67k chars
    transcript = " ".join(words)
    passage = words[5000:5200]
    qt = " ".join(w + "x" if i % 10 == 9 else w for i, w in enumerate(passage))
    started = time.perf_counter()
    span = resolve_llm_quote_span(transcript, qt)
    elapsed = time.perf_counter() - started
    assert span is not None
    assert elapsed < 1.0, f"{elapsed:.2f}s for one 200-word quote"
