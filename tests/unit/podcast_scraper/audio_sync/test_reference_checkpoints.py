"""RFC-129: where an episode's reference puts its checkpoints (pure arithmetic, no audio)."""

from __future__ import annotations

import pytest

from podcast_scraper.audio_sync.reference import ad_regions, checkpoint_times

pytestmark = pytest.mark.unit


def test_one_read_in_pieces_is_one_region() -> None:
    # Per-segment spans, milliseconds, as episode_ad_spans returns them.
    spans = [(6719, 11839), (12139, 16059), (17882, 20702), (600_000, 630_000), (700_000, 701_000)]
    assert ad_regions(spans) == [(6.719, 20.702), (600.0, 630.0), (700.0, 701.0)]


def test_talk_starts_after_the_opening_ads_and_resumes_after_each_break() -> None:
    marks = checkpoint_times(1200.0, [(0.0, 60.0), (500.0, 560.0)])
    # 5 s past where the transcript says the ad ends: talk can resume a few seconds later.
    assert marks[0] == (65.0, "talk_start")
    assert (565.0, "after_ad") in marks
    assert all(not (500.0 <= t < 560.0) for t, _ in marks)  # nothing inside the break


def test_long_talk_gets_a_checkpoint_every_interval() -> None:
    marks = checkpoint_times(1900.0, [])
    assert marks == [
        (0.0, "talk_start"),
        (300.0, "interval"),
        (600.0, "interval"),
        (900.0, "interval"),
        (1200.0, "interval"),
        (1500.0, "interval"),
        (1800.0, "interval"),
    ]


def test_no_window_runs_past_the_end() -> None:
    assert checkpoint_times(5.0, []) == []
    assert all(t <= 590.0 for t, _ in checkpoint_times(600.0, [(580.0, 600.0)]))


def test_an_interval_close_to_a_break_is_dropped() -> None:
    marks = checkpoint_times(1000.0, [(310.0, 320.0)])
    times = [t for t, _ in marks]
    assert (325.0, "after_ad") in marks
    assert all(not (290.0 < t < 320.0) for t in times)
