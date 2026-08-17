"""Name the episodes transcribed from unpreprocessed audio, or the damage stays invisible.

#18/#558: preprocessing ran under a flat 300s budget; on a long episode it hit the wall, produced
nothing, and the ORIGINAL full-size file went to the STT provider. The resulting transcript IS the
artifact, and every downstream artifact derives from it.

Nothing we built repairs that. ``gi-repair`` rebuilds insights FROM the transcript;
``reprocess-corpus-from-transcripts`` runs ``transcribe=off`` by design. So this damage survives
every repair, and until it can be NAMED, "should we re-transcribe?" has no answer — which is how
it stays unfixed.

Measured on real corpora 2026-08-17: pre-#558 9 of 15 runs damaged (60%), all clustered at
297,064-300,845 ms; post-#558 0 of 14, with one run at 324,114 ms that the old flat budget would
have killed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.preprocessing.audit import (
    assess_preprocessing,
    check_corpus_preprocessing,
)

pytestmark = [pytest.mark.unit]


def _run(
    corpus: Path,
    *,
    feed: str = "feed_a",
    run: str = "run_20260815-120000",
    attempts: Optional[int] = 1,
    completed: Optional[int] = 0,
    wall_ms: float = 300_500.0,
    reduction: float = 0.0,
    saved: int = 0,
    episode_ids: Optional[List[str]] = None,
) -> Path:
    run_dir = corpus / "feeds" / feed / run
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Any] = {
        "avg_preprocessing_wall_ms": wall_ms,
        "avg_preprocessing_size_reduction_percent": reduction,
        "total_preprocessing_saved_bytes": saved,
    }
    if attempts is not None:
        metrics["preprocessing_attempts"] = attempts
    if completed is not None:
        metrics["preprocessing_count"] = completed
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    for i, eid in enumerate(episode_ids or ["ep-1"]):
        (run_dir / "metadata" / f"{i:04d} - Episode.metadata.json").write_text(
            json.dumps({"episode": {"episode_id": eid, "title": f"Episode {i}"}}),
            encoding="utf-8",
        )
    return run_dir


def test_attempted_but_never_completed_is_damage(tmp_path):
    """THE signature: preprocessing asked for, produced nothing -> raw audio reached the API."""
    _run(tmp_path, attempts=1, completed=0)

    ok, report = check_corpus_preprocessing(tmp_path)

    assert ok is False
    assert "runs DAMAGED             : 1" in report
    assert "ep-1" in report, "the episode must be named, not just counted"


def test_a_completed_preprocess_is_not_damage(tmp_path):
    _run(tmp_path, attempts=1, completed=1, wall_ms=45_000.0, reduction=75.0, saved=30_000_000)

    ok, report = check_corpus_preprocessing(tmp_path)

    assert ok is True
    assert "VERDICT: PASS" in report


def test_preprocessing_never_attempted_is_not_damage(tmp_path):
    """A cached transcript, or preprocessing switched off — no audio was sent unprocessed."""
    _run(tmp_path, attempts=0, completed=0, wall_ms=0.0)

    ok, _report = check_corpus_preprocessing(tmp_path)

    assert ok is True


def test_the_300s_wall_is_fingerprinted(tmp_path):
    """Distinguishes the #558 flat-budget timeout from any other reason preprocessing produced
    nothing — the tight cluster at ~300s is what identifies the old bug."""
    _run(tmp_path, wall_ms=299_600.0)

    _ok, report = check_corpus_preprocessing(tmp_path)

    assert "of which at the 300s wall: 1" in report


def test_a_slow_failure_far_from_the_wall_is_still_damage_but_not_fingerprinted(tmp_path):
    """Damaged for a different reason: still fails, but must not be blamed on the flat budget."""
    _run(tmp_path, wall_ms=42_000.0)

    ok, report = check_corpus_preprocessing(tmp_path)

    assert ok is False
    assert "runs DAMAGED             : 1" in report
    assert "of which at the 300s wall: 0" in report


def test_a_multi_episode_run_is_reported_as_AMBIGUOUS_not_guessed(tmp_path):
    """metrics.json is RUN-level. With several episodes it cannot say WHICH one was damaged, and
    inventing an attribution would be worse than admitting the limit."""
    _run(tmp_path, episode_ids=["ep-a", "ep-b", "ep-c"])

    ok, report = check_corpus_preprocessing(tmp_path)

    assert ok is False
    assert "cannot say which" in report
    assert "episodes=3" in report


def test_a_single_episode_run_attributes_exactly(tmp_path):
    _run(tmp_path, episode_ids=["only-one"])

    runs = assess_preprocessing(tmp_path)

    assert len(runs) == 1
    assert runs[0].attribution_is_exact is True
    assert runs[0].episode_ids == ["only-one"]


def test_missing_metrics_are_reported_not_silently_passed(tmp_path):
    """A run with no preprocessing metrics cannot be judged; silence would read as 'clean'."""
    _run(tmp_path, attempts=None, completed=None)

    _ok, report = check_corpus_preprocessing(tmp_path)

    assert "runs with no preprocessing metrics at all: 1" in report


def test_an_empty_corpus_says_so(tmp_path):
    ok, report = check_corpus_preprocessing(tmp_path)

    assert ok is True
    assert "no metrics.json found" in report


def test_the_report_states_that_no_repair_we_have_fixes_this(tmp_path):
    """The most important line in the report: this damage is NOT covered by the GI repair."""
    _run(tmp_path)

    _ok, report = check_corpus_preprocessing(tmp_path)

    assert "gi-repair rebuilds insights FROM the transcript" in report
    assert "only repair is re-transcription" in report
