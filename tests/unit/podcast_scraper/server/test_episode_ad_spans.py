"""Where a short listen learns an episode's ads (operator 2026-10-12).

From what is on disk: the segments the pipeline cut (the ad-free file keeps raw start times), else
the ad READS the detector finds — never its whole range, which can sweep in content — and any
segment diarization voiced as ``commercial``.
"""

from __future__ import annotations

from typing import Any

import pytest

from podcast_scraper.server.app_episode_search import ad_spans_from_segments

pytestmark = [pytest.mark.unit]


def _seg(start: float, end: float, text: str, **extra: Any) -> dict[str, Any]:
    return {"start": start, "end": end, "text": text, **extra}


INTRO = "Welcome back to the show. Today we're talking about risk, and I'm joined by a guest."
EPISODE = [
    _seg(0, 6, INTRO),
    _seg(6, 12, "Thanks, great to be back."),
    _seg(12, 18, "This episode is brought to you by Squarespace. Visit squarespace.com/podcast."),
    _seg(18, 24, "What's the one heuristic you keep coming back to?"),
    _seg(24, 30, "Here's how I think about it. Risk lives in the couplings."),
    _seg(30, 36, "Walk me through a concrete example."),
    _seg(36, 42, "It comes down to this: size for the worst week."),
    _seg(42, 48, "We'll be right back after a quick word from our sponsors."),
    _seg(48, 54, "Today's episode is sponsored by Notion. Try notion.com today."),
    *[
        _seg(54 + 6 * i, 60 + 6 * i, f"A long stretch of the conversation, part {i}.")
        for i in range(40)
    ],
]


def test_the_pipeline_cut_is_the_evidence_when_it_is_on_disk() -> None:
    adfree = [s for s in EPISODE if s["start"] not in (12, 42, 48)]
    assert ad_spans_from_segments(EPISODE, adfree) == [(12_000, 18_000), (42_000, 54_000)]


def test_without_it_the_detector_marks_the_reads_not_the_content_between_them() -> None:
    """The detector's range here is 0-54 s; the points at 24 and 36 s are not ads."""
    spans = ad_spans_from_segments(EPISODE, None)
    assert spans == [(12_000, 18_000), (42_000, 54_000)]
    assert not any(a < 30_000 and b > 24_000 for a, b in spans)


def test_an_empty_adfree_file_is_not_an_episode_of_ads() -> None:
    assert ad_spans_from_segments(EPISODE, []) == ad_spans_from_segments(EPISODE, None)


def test_a_read_split_over_three_lines_is_one_span() -> None:
    tail = [
        *EPISODE,
        _seg(300, 303, "Today's episode is brought to you by Stripe."),
        _seg(303, 309, "Stripe simplifies payments for marketplaces."),
        _seg(309, 312, "Visit stripe.com/podcast for a free trial."),
        _seg(312, 318, "That's it for today's episode. See you next time."),
    ]
    assert (300_000, 312_000) in ad_spans_from_segments(tail, None)


def test_a_commercial_voice_is_an_ad_whatever_it_says() -> None:
    segs = [*EPISODE, _seg(400, 410, "A great offer just for listeners.", voice_type="commercial")]
    assert (400_000, 410_000) in ad_spans_from_segments(segs, [s for s in segs])


def test_an_episode_without_ads_has_none() -> None:
    clean = [s for s in EPISODE if s["start"] not in (12, 42, 48)]
    assert ad_spans_from_segments(clean, None) == []
