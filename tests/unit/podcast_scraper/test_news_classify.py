"""The news item classifier against the operator's twelve judgements of 2026-11-19.

Each case stands in for one real teaser the operator judged: invented text with the same shape (its
length band, its markers, the kind of URL), because the publishers' feed text is not ours to put in
a public repository. The real items and their measurement live in the private eval project
(chipi/podcast_scraper#2502).
"""

from __future__ import annotations

import pytest

from podcast_scraper.news_classify import (
    classify,
    DAY_BRIEF,
    is_kept,
    NEWS_SUMMARY,
    NOT_ARTICLE,
    PROMO,
    SMALL_TEASER,
)

pytestmark = pytest.mark.unit


def _words(n: int, lead: str = "") -> str:
    """``lead`` padded with plain reporting words to ``n`` words."""
    filler = "officials said the measure would take effect next year after a long review".split()
    out = lead.split()
    while len(out) < n:
        out.append(filler[len(out) % len(filler)])
    return " ".join(out[:n])


CASES = [
    # (what the operator saw, title, body, url, expected class)
    (
        "a series promotion",
        "A new era for a field",
        _words(
            27,
            "Bold and new, the study has entered a new age. Welcome to our special series on it.",
        ),
        None,
        PROMO,
    ),
    (
        "a status post on a social server",
        "Service not reachable",
        _words(69, "Our domain is unreachable and we are waiting for support."),
        "https://social.example.test/@status/117636437840688845",
        NOT_ARTICLE,
    ),
    (
        "a one-sentence standfirst",
        "Meet the dark bodies",
        _words(23, "Astronomers report a rare pair of bright black holes."),
        None,
        SMALL_TEASER,
    ),
    (
        "the opening of a news report",
        "Leader touts plan at summit",
        _words(78, "The president called for deeper cooperation on trade at the summit."),
        None,
        NEWS_SUMMARY,
    ),
    (
        "a day's brief",
        "Firm opens an outpost",
        _words(83, "Good morning. We have a lot of news today. The need-to-know this morning:"),
        None,
        DAY_BRIEF,
    ),
    (
        "a small page linking two reports",
        "Research Reports",
        _words(
            83,
            "Deep Dive. Artificial intelligence. Models do not reason, a decade after the match.",
        ),
        None,
        NEWS_SUMMARY,
    ),
    ("a stub", "A letter on ocean monitoring", "Letter to the Editor", None, NOT_ARTICLE),
    (
        "a long news opening",
        "Leader greets visitor on arrival",
        _words(228, "The president has landed for the summit, his second trip this year."),
        None,
        NEWS_SUMMARY,
    ),
    (
        "a newsletter's chatter",
        "Policy gains traction",
        _words(55, "The bear added a soft side to his image. Send news tips to the desk."),
        None,
        DAY_BRIEF,
    ),
    (
        "another one-sentence standfirst",
        "A fossil leaves an impression",
        _words(22, "A fossilized print records marks left by a juvenile reptile."),
        None,
        SMALL_TEASER,
    ),
    (
        "a news report's opening",
        "Leader urges cooperation",
        _words(100, "The leader called for a joint effort to safeguard supply chains."),
        None,
        NEWS_SUMMARY,
    ),
    (
        "a paywalled news lead",
        "Startup takes aim at a disease",
        _words(
            96,
            "For years researchers have searched for a treatment. Continue to read the full story.",
        ),
        None,
        NEWS_SUMMARY,
    ),
]


@pytest.mark.parametrize("what, title, body, url, expected", CASES, ids=[c[0] for c in CASES])
def test_the_operators_twelve_judgements(
    what: str, title: str, body: str, url, expected: str
) -> None:
    assert classify(title, body, url) == expected


def test_only_a_news_summary_is_kept() -> None:
    assert is_kept(NEWS_SUMMARY)
    for skipped in (SMALL_TEASER, DAY_BRIEF, PROMO, NOT_ARTICLE, None, ""):
        assert not is_kept(skipped)
