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


def test_a_repaired_episode_stops_being_reported(tmp_path):
    """THE flaw found by the e2e repair test 2026-08-17.

    A run's metrics.json is immutable history. After an episode is re-transcribed into a NEW run
    dir, the OLD run still records attempts=1/completed=0 forever. Counting every run means the
    audit can never go green after a successful repair — the identical flaw the placeholder gate
    had, reintroduced here in a different file.
    """
    _run(
        tmp_path,
        run="run_20260815-120000",
        attempts=1,
        completed=0,
        wall_ms=300_500.0,
        episode_ids=["ep-1"],
    )
    _run(
        tmp_path,
        run="run_20260817-090000",
        attempts=1,
        completed=1,
        wall_ms=45_000.0,
        reduction=80.0,
        saved=50_000_000,
        episode_ids=["ep-1"],
    )

    ok, report = check_corpus_preprocessing(tmp_path)

    assert ok is True, f"the episode the corpus SERVES is now healthy\n{report}"
    assert "runs DAMAGED             : 0" in report


def test_the_superseded_run_is_still_visible_forensically(tmp_path):
    """Scoping is the gate's default, not an erasure — the history stays inspectable."""
    _run(tmp_path, run="run_20260815-120000", attempts=1, completed=0, episode_ids=["ep-1"])
    _run(
        tmp_path,
        run="run_20260817-090000",
        attempts=1,
        completed=1,
        wall_ms=45_000.0,
        reduction=80.0,
        episode_ids=["ep-1"],
    )

    current = assess_preprocessing(tmp_path)
    everything = assess_preprocessing(tmp_path, current_only=False)

    assert len(current) == 1, "the gate judges only the served copy"
    assert len(everything) == 2, "forensics can still see the superseded run"
    assert any(r.damaged for r in everything)


def test_an_episode_never_repaired_is_still_reported(tmp_path):
    """The scoping must not become a way for real damage to disappear."""
    _run(tmp_path, run="run_20260815-120000", attempts=1, completed=0, episode_ids=["ep-broken"])

    ok, report = check_corpus_preprocessing(tmp_path)

    assert ok is False
    assert "ep-broken" in report


def test_a_PARTIALLY_failed_batch_run_is_damage(tmp_path):
    """The rule was `completed == 0`, which is only correct when one run means one episode.

    A production run of 50 episodes where 45 preprocessed and 5 hit the wall records
    attempts=50/completed=45 and read as HEALTHY under the old rule — hiding all 5. My local
    corpora were one-episode-per-run, so the degenerate case was the only one ever validated.
    """
    _run(
        tmp_path,
        attempts=50,
        completed=45,
        wall_ms=120_000.0,
        episode_ids=[f"ep-{i}" for i in range(50)],
    )

    ok, report = check_corpus_preprocessing(tmp_path)

    assert ok is False, f"5 of 50 episodes went to the provider unpreprocessed\n{report}"
    assert "runs DAMAGED             : 1" in report


def test_a_fully_successful_batch_run_is_not_damage(tmp_path):
    _run(
        tmp_path,
        attempts=50,
        completed=50,
        wall_ms=120_000.0,
        reduction=75.0,
        episode_ids=[f"ep-{i}" for i in range(50)],
    )

    ok, _report = check_corpus_preprocessing(tmp_path)

    assert ok is True


def test_partial_damage_is_distinguishable_from_total(tmp_path):
    """Changes what an operator can conclude: in a partial run NO episode can be cleared."""
    _run(tmp_path, run="run_a", attempts=10, completed=7, episode_ids=[f"a{i}" for i in range(10)])
    _run(tmp_path, run="run_b", attempts=3, completed=0, episode_ids=[f"b{i}" for i in range(3)])

    runs = {Path(r.run_dir).name: r for r in assess_preprocessing(tmp_path)}

    assert runs["run_a"].partially_damaged is True
    assert runs["run_b"].partially_damaged is False, "a total failure is not a partial one"
    assert runs["run_a"].damaged and runs["run_b"].damaged
