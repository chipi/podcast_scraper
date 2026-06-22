"""Unit tests for gi.position_hint — RFC-097 4-step waterfall.

Verifies each rung of the waterfall in isolation:

1. RSS ``episode_duration_ms``
2. Last segment's ``end`` × 1000 (NEW in RFC-097 chunk 5)
3. ``max(Quote.timestamp_end_ms)`` fallback
4. Skip emission (None)
"""

from __future__ import annotations

import pytest

from podcast_scraper.gi.position_hint import compute_position_hint

pytestmark = pytest.mark.unit


# ─── Step 1: RSS duration ───


def test_step1_rss_duration_used_when_present():
    """Step 1 wins when episode_duration_ms is set."""
    value, step = compute_position_hint(
        timestamp_starts_ms=[15000, 25000],
        episode_duration_ms=100000,
    )
    assert step == 1
    assert value == pytest.approx(0.20)  # mean(15000, 25000) = 20000; 20000/100000


def test_step1_clamped_to_one():
    """Quote start beyond duration clamps to 1.0 (no >1 ratios)."""
    value, step = compute_position_hint(
        timestamp_starts_ms=[200000],
        episode_duration_ms=100000,
    )
    assert step == 1
    assert value == 1.0


# ─── Step 2: segments fallback (new in chunk 5) ───


def test_step2_segments_end_used_when_rss_missing():
    """Step 2: last segment's ``end`` × 1000 used when episode_duration_ms is None."""
    segments = [
        {"start": 0.0, "end": 10.0, "text": "a"},
        {"start": 10.0, "end": 60.0, "text": "b"},
        {"start": 60.0, "end": 200.0, "text": "c"},
    ]
    value, step = compute_position_hint(
        timestamp_starts_ms=[50000],
        episode_duration_ms=None,
        transcript_segments=segments,
    )
    assert step == 2
    # 50000 ms / (200 s × 1000) = 0.25
    assert value == pytest.approx(0.25)


def test_step2_handles_object_shaped_segments():
    """Works with dataclass/object segments (``.end`` attr)."""

    class Seg:
        def __init__(self, end):
            self.end = end

    segments = [Seg(end=10.0), Seg(end=100.0)]
    value, step = compute_position_hint(
        timestamp_starts_ms=[25000],
        episode_duration_ms=None,
        transcript_segments=segments,
    )
    assert step == 2
    assert value == pytest.approx(0.25)


def test_step2_segments_zero_or_negative_skipped():
    """Last segment end ≤ 0 falls through to step 3."""
    segments = [{"start": 0.0, "end": 0.0}]
    value, step = compute_position_hint(
        timestamp_starts_ms=[5000],
        episode_duration_ms=None,
        transcript_segments=segments,
        quote_end_fallback_ms=10000,
    )
    assert step == 3
    assert value == pytest.approx(0.5)


def test_step2_invalid_segment_end_falls_through():
    """Non-numeric ``end`` falls through to step 3 / 4."""
    segments = [{"start": 0.0, "end": "not-a-number"}]
    value, step = compute_position_hint(
        timestamp_starts_ms=[1000],
        episode_duration_ms=None,
        transcript_segments=segments,
        quote_end_fallback_ms=2000,
    )
    assert step == 3
    assert value == pytest.approx(0.5)


# ─── Step 3: max Quote timestamp_end_ms ───


def test_step3_quote_end_fallback_used_when_no_segments():
    """Step 3 (max quote end) used when RSS duration + segments both unavailable."""
    value, step = compute_position_hint(
        timestamp_starts_ms=[3000],
        episode_duration_ms=None,
        transcript_segments=None,
        quote_end_fallback_ms=12000,
    )
    assert step == 3
    assert value == pytest.approx(0.25)


# ─── Step 4: skip emission ───


def test_step4_returns_none_when_all_sources_missing():
    """Step 4: no duration recoverable from any source → return None."""
    value, step = compute_position_hint(
        timestamp_starts_ms=[1000, 2000],
        episode_duration_ms=None,
        transcript_segments=None,
        quote_end_fallback_ms=None,
    )
    assert value is None
    assert step == 4


def test_step4_returns_none_when_no_quote_starts():
    """Empty quote starts → step 4 regardless of duration sources."""
    value, step = compute_position_hint(
        timestamp_starts_ms=[],
        episode_duration_ms=100000,
        transcript_segments=[{"end": 200.0}],
        quote_end_fallback_ms=50000,
    )
    assert value is None
    assert step == 4


# ─── Waterfall ordering ───


def test_waterfall_priority_step1_beats_step2():
    """Step 1 (RSS) wins over step 2 (segments) when both present."""
    segments = [{"start": 0.0, "end": 1000.0}]  # 1,000,000 ms
    value, step = compute_position_hint(
        timestamp_starts_ms=[50000],
        episode_duration_ms=100000,  # 100,000 ms — different from segments
        transcript_segments=segments,
    )
    assert step == 1  # RSS wins
    assert value == pytest.approx(0.5)  # 50000 / 100000


def test_waterfall_priority_step2_beats_step3():
    """Step 2 (segments) wins over step 3 (max quote end) when both present."""
    segments = [{"start": 0.0, "end": 100.0}]  # 100,000 ms
    value, step = compute_position_hint(
        timestamp_starts_ms=[25000],
        episode_duration_ms=None,
        transcript_segments=segments,
        quote_end_fallback_ms=50000,  # would yield 0.5; segments yields 0.25
    )
    assert step == 2
    assert value == pytest.approx(0.25)
