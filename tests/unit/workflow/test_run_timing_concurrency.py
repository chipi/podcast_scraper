"""run_timing must not report a wall-clock share it cannot honour (2026-08-31).

The pipeline processes episodes on a pool (``processing=2`` by default), so per-stage
seconds OVERLAP on a multi-episode run and their sum can exceed the run's wall-clock.
Emitting that ratio as a percentage produced ``model_stage_share_pct: 203.1`` on prod — a
number that cannot be true, on the very event added to answer "where does the time go".

A metric that reports an impossible value is worse than one that reports nothing: it gets
quoted. The share is now emitted ONLY when it is exact (single-episode runs), and the
concurrency-honest form is always available.
"""

from __future__ import annotations

import json

from podcast_scraper.workflow import metrics as metrics_mod
from podcast_scraper.workflow.orchestration import _emit_run_timing_event


def _emit(caplog, episodes: int, *, stage_sec_each: float = 500.0, wall: float = 600.0) -> dict:
    caplog.set_level("INFO")
    caplog.clear()
    m = metrics_mod.Metrics()
    m.episodes_scraped_total = episodes
    per = stage_sec_each / 5.0
    m.transcribe_time_by_episode = {i: per for i in range(max(1, episodes))}
    m.summarize_time_by_episode = {i: per * 4 for i in range(max(1, episodes))}
    m._start_time -= wall

    _emit_run_timing_event(m)
    for rec in caplog.records:
        msg = rec.getMessage()
        if '"run_timing"' in msg:
            parsed: dict = json.loads(msg)
            return parsed
    raise AssertionError("run_timing not emitted")


def test_single_episode_run_reports_an_exact_share(caplog):
    ev = _emit(caplog, 1)
    assert ev["episodes_in_run"] == 1
    assert ev["model_stage_share_pct"] is not None
    assert 0 < ev["model_stage_share_pct"] <= 100


def test_multi_episode_run_omits_the_share_entirely(caplog):
    """THE REGRESSION: this used to emit 203.1%."""
    ev = _emit(caplog, 8)
    assert ev["episodes_in_run"] == 8
    assert (
        "model_stage_share_pct" not in ev
    ), f"share_pct must be omitted for concurrent runs, got {ev.get('model_stage_share_pct')}"


def test_share_is_never_above_100_when_present(caplog):
    """Whatever the shape, a percentage that is emitted must be a real percentage."""
    for eps in (0, 1, 2, 5, 8, 20):
        ev = _emit(caplog, eps)
        pct = ev.get("model_stage_share_pct")
        if pct is not None:
            assert pct <= 100.0, f"episodes={eps} produced an impossible share {pct}"


def test_concurrency_is_reported_instead_and_is_interpretable(caplog):
    """8 episodes x 500s of stages in 600s wall == ~6.7 streams in flight, not 667%."""
    ev = _emit(caplog, 8)
    assert ev["mean_model_stage_concurrency"] > 1.0
    expected = ev["model_stage_sec_total"] / ev["run_duration_sec"]
    assert abs(ev["mean_model_stage_concurrency"] - expected) < 0.01


def test_per_episode_cost_is_comparable_across_run_shapes(caplog):
    """The number that lets a 1-episode run and an 8-episode run be compared at all."""
    one = _emit(caplog, 1)
    eight = _emit(caplog, 8)
    assert (
        one["model_stage_sec_per_selected_episode"] == eight["model_stage_sec_per_selected_episode"]
    )


def test_unaccounted_is_suppressed_when_it_would_be_meaningless(caplog):
    """run - model - local is only a residual if the stages did not overlap."""
    assert _emit(caplog, 1).get("unaccounted_sec") is not None
    assert "unaccounted_sec" not in _emit(caplog, 8)
