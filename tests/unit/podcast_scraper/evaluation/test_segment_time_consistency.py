"""Unit tests for the transcript segment-time consistency validator (#1173, AC3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.evaluation.segment_time_consistency import (
    compute_segment_time_consistency,
    discover_segment_files,
    enforce_segment_time_consistency,
)

pytestmark = pytest.mark.unit


def _write_segments(root: Path, stem: str, segments: list[dict], *, adfree: bool = False) -> Path:
    d = root / "feeds" / "f1" / "run_20260101-000000" / "transcripts"
    d.mkdir(parents=True, exist_ok=True)
    suffix = ".adfree.segments.json" if adfree else ".segments.json"
    path = d / f"{stem}{suffix}"
    path.write_text(json.dumps(segments), encoding="utf-8")
    return path


def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


def test_clean_corpus_passes(tmp_path: Path) -> None:
    # ~2.5 chars/sec — normal speech, comfortably word-accurate.
    _write_segments(
        tmp_path,
        "clean",
        [
            _seg(0.0, 4.0, "Welcome to the show today everyone."),
            _seg(4.0, 9.0, "We are going to talk about the economy at length."),
            _seg(9.0, 12.0, "It has been a strange year so far."),
        ],
    )
    metrics = compute_segment_time_consistency(tmp_path)
    assert metrics["episodes_total"] == 1
    assert metrics["impossible_segments"] == 0
    passed, failures = enforce_segment_time_consistency(metrics)
    assert passed, failures


def test_impossible_speech_rate_fails(tmp_path: Path) -> None:
    # 21 chars in 0.2s = 105 cps — the real prod-v2 symptom (coarse/wrong segment times).
    _write_segments(
        tmp_path,
        "drift",
        [
            _seg(0.0, 5.0, "A perfectly normal opening line for the episode."),
            _seg(5.0, 5.2, "in the United States."),
        ],
    )
    metrics = compute_segment_time_consistency(tmp_path)
    assert metrics["impossible_segments"] == 1
    assert metrics["episodes_with_impossible"] == 1
    passed, failures = enforce_segment_time_consistency(metrics)
    assert not passed
    assert any("impossible" in f for f in failures)


def test_nonmonotonic_and_out_of_bounds_fail(tmp_path: Path) -> None:
    # Order isolates each anomaly: negative-start first (out-of-bounds, still monotonic vs the
    # -1.0 sentinel), then a backwards jump (non-monotonic only).
    _write_segments(
        tmp_path,
        "bad",
        [
            _seg(-1.0, 2.0, "This one has a negative start time and is out of bounds."),
            _seg(10.0, 12.0, "This segment starts at ten seconds in."),
            _seg(3.0, 5.0, "But this one jumps backwards to three seconds."),  # non-monotonic
        ],
    )
    metrics = compute_segment_time_consistency(tmp_path)
    assert metrics["nonmonotonic_starts"] == 1
    assert metrics["out_of_bounds"] == 1
    passed, failures = enforce_segment_time_consistency(metrics)
    assert not passed
    assert len(failures) == 2


def test_short_segments_do_not_trip_rate(tmp_path: Path) -> None:
    # 1-2 word backchannels have meaningless rates; must not be scored as impossible.
    _write_segments(
        tmp_path,
        "short",
        [
            _seg(0.0, 3.0, "Right, exactly, that is the whole point here."),
            _seg(3.0, 3.05, "Yeah."),
            _seg(3.05, 3.1, "Mm-hmm."),
        ],
    )
    metrics = compute_segment_time_consistency(tmp_path)
    assert metrics["impossible_segments"] == 0
    assert enforce_segment_time_consistency(metrics)[0]


def test_adfree_segments_excluded_from_discovery(tmp_path: Path) -> None:
    _write_segments(tmp_path, "ep", [_seg(0.0, 4.0, "The raw provider segment output.")])
    _write_segments(tmp_path, "ep", [_seg(0.0, 4.0, "The ad-free re-slice.")], adfree=True)
    found = discover_segment_files(tmp_path)
    assert len(found) == 1
    assert found[0].name == "ep.segments.json"
