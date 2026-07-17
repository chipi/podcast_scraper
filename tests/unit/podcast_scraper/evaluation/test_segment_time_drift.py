"""Unit tests for the segment-time drift alignment/metric math (#1173, AC1/AC2).

Pure-math tests over synthetic word streams — no whisper. The live fixture-transcription harness
and the real-fixture regression assertion live in tests/integration/eval.
"""

from __future__ import annotations

import pytest

from podcast_scraper.evaluation.segment_time_drift import (
    align_word_streams,
    measure_boundary_drift,
    percentile,
    pool_drift,
    refined_and_segment_timelines,
)

pytestmark = pytest.mark.unit


def test_align_tolerates_insert_delete_substitute() -> None:
    source = ["the", "quick", "brown", "fox", "jumps"]
    # transcript drops "quick", garbles "brown"->"braun", inserts "very".
    transcript = ["the", "braun", "fox", "very", "jumps"]
    mapping = align_word_streams(source, transcript)
    # "the" (0->0), "fox" (3->2), "jumps" (4->4) survive; dropped/garbled words are unmapped.
    assert mapping[0] == 0
    assert mapping[3] == 2
    assert mapping[4] == 4
    assert 1 not in mapping  # "quick" deleted
    assert 2 not in mapping  # "brown" garbled


def test_percentile_nearest_rank() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([10, 20, 30, 40, 50], 50) == 30
    assert percentile([10, 20, 30, 40, 50], 100) == 50


def test_boundary_drift_matches_word_times() -> None:
    # turn0 = "hello world", turn1 = "goodbye now"; turn1's true onset is 5.0s.
    source = [
        ("hello", 0, True),
        ("world", 0, False),
        ("goodbye", 1, True),
        ("now", 1, False),
    ]
    onsets = [0.0, 5.0]
    transcript = [("hello", 0.0), ("world", 2.0), ("goodbye", 5.05), ("now", 7.0)]
    result = measure_boundary_drift(source, transcript, onsets)
    assert result.boundaries_total == 1  # only turn1 is a scored boundary (turn0 skipped)
    assert result.boundaries_matched == 1
    assert result.max_ms == pytest.approx(50.0, abs=1e-6)  # |5.05 - 5.0| * 1000


def test_unmatched_boundary_not_scored() -> None:
    # turn1's first word is garbled in the transcript → boundary counted but not scored.
    source = [("intro", 0, True), ("kickoff", 1, True), ("tail", 1, False)]
    onsets = [0.0, 4.0]
    transcript = [("intro", 0.0), ("kikoff", 4.2), ("tail", 4.5)]  # "kickoff" garbled
    result = measure_boundary_drift(source, transcript, onsets)
    assert result.boundaries_total == 1
    assert result.boundaries_matched == 0
    assert result.drifts_ms == []


def test_refined_beats_unrefined_on_drift() -> None:
    # The boundary word "goodbye" lands in a segment that started far earlier (coarse seg start).
    segments = [
        {"start": 0.0, "words": [["hello", 0.0], ["world", 2.0], ["goodbye", 5.05]]},
        {"start": 6.5, "words": [["now", 7.0]]},
    ]
    source = [
        ("hello", 0, True),
        ("world", 0, False),
        ("goodbye", 1, True),
        ("now", 1, False),
    ]
    onsets = [0.0, 5.0]
    refined, unrefined = refined_and_segment_timelines(segments)
    refined_drift = measure_boundary_drift(source, refined, onsets)
    unrefined_drift = measure_boundary_drift(source, unrefined, onsets)
    # Refined (word time 5.05) is ~50ms off; unrefined (segment start 0.0) is ~5s off.
    assert refined_drift.max_ms == pytest.approx(50.0, abs=1e-6)
    assert unrefined_drift.max_ms == pytest.approx(5000.0, abs=1e-6)
    assert refined_drift.max_ms < unrefined_drift.max_ms


def test_pool_drift_aggregates() -> None:
    a = measure_boundary_drift(
        [("x", 0, True), ("y", 1, True)], [("x", 0.0), ("y", 1.1)], [0.0, 1.0]
    )
    b = measure_boundary_drift(
        [("x", 0, True), ("y", 1, True)], [("x", 0.0), ("y", 2.3)], [0.0, 2.0]
    )
    pooled = pool_drift([a, b])
    assert pooled["boundaries"] == 2
    assert pooled["max_ms"] == pytest.approx(300.0, abs=1e-6)
