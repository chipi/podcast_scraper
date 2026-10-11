"""Why a recommendation is here (ADR-163): the one reason shape every Discover rail carries.

Extractive only: a reason's quote is a real quote with where it is said, or nothing.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from podcast_scraper.server.app_reasons import (
    best_grounded_quote,
    cut_quote,
    grounded_quote,
    insights_by_episode,
    QUOTE_MAX_CHARS,
    quote_support,
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
        Reason(kind="another_view", quote=None),
    ]
    for r in reasons:
        assert AppReason.model_validate(r.to_dict()).kind == r.kind
    assert AppReason.model_validate(reasons[-1].to_dict()).quote is None


def _ins(iid: str, text: str, *quotes: AppQuote, slug: str = "ep-1") -> NS:
    return NS(id=iid, text=text, quotes=list(quotes), episode_slug=slug)


def test_a_greeting_never_beats_a_quote_that_backs_the_insight() -> None:
    """The fixture's shape: a greeting heads the insight's quote list; the backing quote follows."""
    q = best_grounded_quote(
        [
            _ins(
                "i1",
                "Risk lives in the couplings between systems, not in the parts.",
                _q("Thanks, Sam. Great to be back.", 1_000, "Elena"),
                _q("Risk lives in the couplings between systems.", 9_000, "Elena"),
            )
        ],
        "ep-1",
    )
    assert q is not None
    assert (q.text, q.start_ms, q.insight_id) == (
        "Risk lives in the couplings between systems.",
        9_000,
        "i1",
    )


def test_a_quote_filed_under_the_neighbouring_insight_is_credited_to_the_one_it_backs() -> None:
    q = best_grounded_quote(
        [
            _ins("i1", "Before we start, a word on last week.", _q("Sponsored by Notion.", 5)),
            _ins("i2", "Labor data is noisier than I argued.", _q("Before we start.", 7)),
            _ins("i3", "Two years ago I argued labor data was reliable.", _q("Real talk.", 9)),
            _ins("i4", "x", _q("Two years ago I argued labor data was reliable quarterly.", 11)),
        ],
        "ep-1",
    )
    assert q is not None and q.insight_id == "i3" and q.start_ms == 11


def test_with_nothing_backing_any_insight_the_first_quote_still_grounds_it() -> None:
    """No card is lost to the support measure: it only reorders, it never empties."""
    q = best_grounded_quote([_ins("i1", "Macro is risk management.", _q("Thanks, Alex.", 3))], "ep")
    assert q is not None and (q.text, q.insight_id) == ("Thanks, Alex.", "i1")
    assert best_grounded_quote([_ins("i1", "anything")], "ep") is None


def test_said_by_keeps_to_one_speakers_quotes() -> None:
    insight = _ins(
        "i1",
        "Diversification is the closest thing to a free lunch.",
        _q("Diversification is the closest thing to a free lunch.", 1, "Host"),
        _q("Diversification, yes, the closest thing to a free lunch.", 2, "Daniel"),
    )
    q = best_grounded_quote([insight], "ep", said_by="Daniel", speaker="Daniel Cho")
    assert q is not None and (q.start_ms, q.speaker) == (2, "Daniel Cho")


def test_support_counts_the_quotes_content_words_the_insight_shares() -> None:
    assert quote_support("Thanks, Sam. Great to be back.", "Risk lives in couplings") == 0.0
    assert quote_support("Risk lives in couplings", "risk LIVES somewhere") == 2 / 3
    assert quote_support("ok so", "anything") == 0.0  # no content words


def test_insights_group_by_episode_in_first_seen_order() -> None:
    a, b, c = _ins("1", "", slug="ep-b"), _ins("2", "", slug="ep-a"), _ins("3", "", slug="ep-b")
    assert [
        (s, [i.id for i in g]) for s, g in insights_by_episode([a, b, c, _ins("4", "", slug="")])
    ] == [
        ("ep-b", ["1", "3"]),
        ("ep-a", ["2"]),
    ]
