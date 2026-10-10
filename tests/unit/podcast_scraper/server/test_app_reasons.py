"""Why a recommendation is here (ADR-163): the one reason shape every Discover rail carries.

Extractive only: a reason's quote is a real quote with where it is said, or nothing.
"""

from __future__ import annotations

import pytest

from podcast_scraper.server.app_reasons import (
    cut_quote,
    grounded_quote,
    QUOTE_MAX_CHARS,
    Reason,
    ReasonCapture,
    ReasonSubject,
)
from podcast_scraper.server.schemas import AppQuote, AppReason

pytestmark = [pytest.mark.unit]


def _q(text: str, start_ms: int | None = None, speaker: str | None = None) -> AppQuote:
    return AppQuote(text=text, speaker=speaker, start_ms=start_ms, end_ms=None)


def test_a_timed_quote_grounds_the_reason_before_an_untimed_one() -> None:
    q = grounded_quote([_q("untimed"), _q("timed", 61_000, "Ann")], "ep-1", insight_id="i1")
    assert q is not None
    assert (q.text, q.start_ms, q.speaker, q.episode_slug, q.insight_id) == (
        "timed",
        61_000,
        "Ann",
        "ep-1",
        "i1",
    )


def test_no_quote_text_means_no_quote_never_a_substitute() -> None:
    assert grounded_quote([], "ep-1") is None
    assert grounded_quote([_q("   ")], "ep-1") is None


def test_an_untimed_quote_still_grounds_it_without_a_time() -> None:
    q = grounded_quote([_q("said, time unknown")], "ep-1")
    assert q is not None and q.start_ms is None


def test_the_named_speaker_wins_over_the_quotes_own() -> None:
    q = grounded_quote([_q("x", 1, speaker=None)], "ep-1", speaker="Bo")
    assert q is not None and q.speaker == "Bo"


def test_long_quotes_are_cut_for_a_card() -> None:
    cut = cut_quote("word " * 200)
    assert len(cut) == QUOTE_MAX_CHARS and cut.endswith("…")
    assert cut_quote("  a \n b  ") == "a b"


def test_every_reason_fits_the_api_shape() -> None:
    quote = grounded_quote([_q("Said it.", 9_000)], "ep-1")
    reasons = [
        Reason(
            kind="rising_topic",
            quote=quote,
            subject=ReasonSubject(kind="topic", id="topic:ai", label="AI"),
        ),
        Reason(kind="most_engaged", quote=quote, events=4),
        Reason(
            kind="captured",
            quote=quote,
            subject=ReasonSubject(kind="person", id="person:ann", label="Ann"),
            capture=ReasonCapture(kind="highlight", episode_slug="ep-0", text="mine", start_ms=1),
        ),
        Reason(kind="thread", subject=ReasonSubject(kind="storyline", id="thc:1", label="S")),
        Reason(kind="another_view", quote=None, outside_habits=True),
    ]
    for r in reasons:
        assert AppReason.model_validate(r.to_dict()).kind == r.kind
    assert AppReason.model_validate(reasons[-1].to_dict()).quote is None
