"""Tests for the parallelism observability additions to ``Metrics`` (#1180)."""

from __future__ import annotations

import logging

import pytest

from podcast_scraper.workflow.metrics import (
    _intervals_intersection_total,
    _merge_intervals,
    Metrics,
)

pytestmark = pytest.mark.unit


# ---------- interval-math helpers ------------------------------------------


def test_merge_intervals_empty() -> None:
    assert _merge_intervals([]) == []


def test_merge_intervals_disjoint_preserved_and_sorted() -> None:
    assert _merge_intervals([(5, 7), (1, 3)]) == [(1, 3), (5, 7)]


def test_merge_intervals_overlapping_merged() -> None:
    assert _merge_intervals([(1, 3), (2, 4)]) == [(1, 4)]


def test_merge_intervals_touching_merged() -> None:
    assert _merge_intervals([(1, 3), (3, 5)]) == [(1, 5)]


def test_merge_intervals_nested_merged() -> None:
    assert _merge_intervals([(1, 10), (3, 5)]) == [(1, 10)]


def test_intersection_full_containment() -> None:
    assert _intervals_intersection_total([(1, 5)], [(2, 4)]) == 2


def test_intersection_disjoint() -> None:
    assert _intervals_intersection_total([(1, 3)], [(4, 5)]) == 0


def test_intersection_partial_two_intervals_each() -> None:
    # a covers 1-3 and 5-7; b covers 2-6 → 1 + 1 = 2 seconds of overlap.
    assert _intervals_intersection_total([(1, 3), (5, 7)], [(2, 6)]) == 2


def test_intersection_touching_zero() -> None:
    # Adjacent-touch intervals have zero-length intersection.
    assert _intervals_intersection_total([(1, 3)], [(3, 5)]) == 0


# ---------- Metrics recording contract -------------------------------------


def test_record_transcription_active_ignores_reverse_intervals() -> None:
    """A `record_*_active(start, end)` where end <= start is silently dropped —
    it means a caller mis-timed the interval; we don't want it corrupting
    the overlap ratio.
    """
    m = Metrics()
    m.record_transcription_thread_active(10.0, 10.0)  # zero-length
    m.record_transcription_thread_active(10.0, 5.0)  # reversed
    assert m.transcription_thread_intervals == []


def test_record_handoff_latency_ignores_negative() -> None:
    m = Metrics()
    m.record_handoff_latency(-0.1)
    m.record_handoff_latency(0.0)
    m.record_handoff_latency(1.5)
    assert m.handoff_latency_seconds_per_episode == [0.0, 1.5]


def test_processing_queue_idle_ignores_non_positive() -> None:
    m = Metrics()
    m.record_processing_queue_idle_time(0.0)
    m.record_processing_queue_idle_time(-1.0)
    m.record_processing_queue_idle_time(3.5)
    assert m.processing_thread_queue_idle_seconds == 3.5


# ---------- finalize_parallelism_snapshot semantics ------------------------


def test_finalize_perfect_overlap_is_1() -> None:
    m = Metrics()
    m.record_transcription_thread_active(0.0, 10.0)
    m.record_processing_thread_active(0.0, 10.0)
    m.finalize_parallelism_snapshot(pipeline_wall_seconds=10.0)
    assert m.processing_overlap_ratio == 1.0
    assert m.processing_thread_busy_ratio == 1.0


def test_finalize_fully_serial_is_0() -> None:
    m = Metrics()
    m.record_transcription_thread_active(0.0, 10.0)
    m.record_processing_thread_active(10.0, 20.0)
    m.finalize_parallelism_snapshot(pipeline_wall_seconds=20.0)
    assert m.processing_overlap_ratio == 0.0
    assert m.processing_thread_busy_ratio == 0.5


def test_finalize_half_overlap() -> None:
    m = Metrics()
    m.record_transcription_thread_active(0.0, 10.0)
    m.record_processing_thread_active(5.0, 15.0)
    m.finalize_parallelism_snapshot(pipeline_wall_seconds=15.0)
    # tx window 0..10, proc window 5..15 → overlap 5..10 = 5s / 10s tx = 0.5
    assert m.processing_overlap_ratio == 0.5


def test_finalize_ratio_is_none_when_no_transcription_recorded() -> None:
    """A run with no transcription (e.g. dry-run) yields overlap_ratio None —
    a real 0.0 would misleadingly suggest "we tried and failed to overlap".
    """
    m = Metrics()
    m.finalize_parallelism_snapshot(pipeline_wall_seconds=5.0)
    assert m.processing_overlap_ratio is None
    assert m.processing_thread_busy_ratio is None


def test_finalize_is_idempotent() -> None:
    m = Metrics()
    m.record_transcription_thread_active(0.0, 10.0)
    m.record_processing_thread_active(5.0, 15.0)
    m.finalize_parallelism_snapshot(pipeline_wall_seconds=15.0)
    first = (m.processing_overlap_ratio, m.processing_thread_busy_ratio)
    m.finalize_parallelism_snapshot(pipeline_wall_seconds=15.0)
    second = (m.processing_overlap_ratio, m.processing_thread_busy_ratio)
    assert first == second


def test_finalize_infers_wall_when_missing() -> None:
    """When pipeline_wall_seconds isn't provided, the widest span across
    recorded intervals is the denominator.
    """
    m = Metrics()
    m.record_transcription_thread_active(2.0, 8.0)
    m.record_processing_thread_active(3.0, 12.0)
    m.finalize_parallelism_snapshot()
    # widest span = min(starts)=2, max(ends)=12 → wall=10
    # proc active = 9 → busy_ratio = 0.9
    assert m.processing_thread_busy_ratio == 0.9


def test_finalize_warns_when_safety_net_fires_with_inline_active(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If ANY episode was processed by the safety net despite the inline
    ProcessingProcessor being enabled, we log a warning — that's the
    "inline path silently skipping episodes" signal the operator wanted.
    """
    m = Metrics()
    m.record_inline_processed_episode()
    m.record_safety_net_processed_episode()

    caplog.set_level(logging.WARNING, logger="podcast_scraper.workflow.metrics")
    m.finalize_parallelism_snapshot()

    assert any(
        "safety net" in rec.message and rec.levelno == logging.WARNING for rec in caplog.records
    ), f"expected warning about safety-net episodes, got: {[r.message for r in caplog.records]}"


def test_finalize_does_not_warn_when_inline_zero(caplog: pytest.LogCaptureFixture) -> None:
    """Runs with no inline processing (disabled, dry-run) should not fire the
    warning even if safety-net > 0 — the warning is specifically about
    silent skipping when inline was enabled.
    """
    m = Metrics()
    m.record_safety_net_processed_episode()

    caplog.set_level(logging.WARNING, logger="podcast_scraper.workflow.metrics")
    m.finalize_parallelism_snapshot()

    assert not any(
        "safety net" in rec.message for rec in caplog.records
    ), "should not warn when inline processing was never used"


# ---------- Export surface --------------------------------------------------


def test_finish_exports_parallelism_keys() -> None:
    """Every #1180 field lands in the export dict, even on an empty run."""
    m = Metrics()
    d = m.finish()
    for key in (
        "processing_overlap_ratio",
        "processing_thread_busy_ratio",
        "processing_thread_queue_idle_seconds",
        "inline_processed_episodes_count",
        "safety_net_processed_episodes_count",
        "handoff_latency_seconds_per_episode",
    ):
        assert key in d, f"missing {key!r} from finish() output"


def test_finish_ratios_are_none_on_empty_run() -> None:
    d = Metrics().finish()
    assert d["processing_overlap_ratio"] is None
    assert d["processing_thread_busy_ratio"] is None
    assert d["inline_processed_episodes_count"] == 0
    assert d["safety_net_processed_episodes_count"] == 0
    assert d["processing_thread_queue_idle_seconds"] == 0.0
    assert d["handoff_latency_seconds_per_episode"] == []


def test_finish_rounds_handoff_latencies() -> None:
    m = Metrics()
    m.record_handoff_latency(0.123456)
    m.record_handoff_latency(1.9999)
    d = m.finish()
    assert d["handoff_latency_seconds_per_episode"] == [0.123, 2.0]


def test_validate_metrics_requires_new_keys() -> None:
    """The validation gate must know about the new keys so future refactors
    that drop them from the export fail loudly.
    """
    m = Metrics()
    d = m.finish()
    # If validation raises, one of the keys is missing.
    m._validate_metrics(d)


# ---------- log_parallelism_summary (#1180 gap 4) ---------------------------


def test_log_parallelism_summary_emits_info_line_with_all_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The one-liner emits at INFO with every field labelled."""
    m = Metrics()
    m.record_transcription_thread_active(0.0, 10.0)
    m.record_processing_thread_active(5.0, 15.0)
    m.record_inline_processed_episode()
    m.record_inline_processed_episode()
    m.record_safety_net_processed_episode()
    m.record_processing_queue_idle_time(3.5)
    m.record_handoff_latency(0.1)
    m.record_handoff_latency(0.5)
    m.record_handoff_latency(1.2)
    m.finalize_parallelism_snapshot(pipeline_wall_seconds=15.0)

    caplog.set_level(logging.INFO, logger="podcast_scraper.workflow.metrics")
    m.log_parallelism_summary()

    matches = [rec for rec in caplog.records if rec.message.startswith("Parallelism:")]
    assert len(matches) == 1, f"expected exactly one 'Parallelism:' line, got {caplog.records}"
    line = matches[0].message
    assert "overlap=0.50" in line
    assert "busy=" in line
    assert "queue_idle=3.5s" in line
    assert "inline=2" in line
    assert "safety_net=1" in line
    # 3 samples → p95_idx = round(0.95 * 2) = 2 → sorted[2] = 1.2s.
    assert "handoff_p95=1.20s" in line


def test_log_parallelism_summary_uses_na_when_no_threads_engaged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dry-run / disabled: ratios read `n/a`, not misleading 0.00."""
    m = Metrics()
    m.finalize_parallelism_snapshot()
    caplog.set_level(logging.INFO, logger="podcast_scraper.workflow.metrics")
    m.log_parallelism_summary()

    matches = [rec for rec in caplog.records if rec.message.startswith("Parallelism:")]
    assert matches
    line = matches[0].message
    assert "overlap=n/a" in line
    assert "busy=n/a" in line
    assert "handoff_p95=n/a" in line


def test_pipeline_status_json_payload_does_not_carry_parallelism_keys(
    tmp_path: object,
) -> None:
    """`.pipeline_status.json` is a per-stage progress tracker
    (`monitor/status.py::maybe_update_pipeline_status`), NOT a metrics dump.
    The #1180 parallelism fields live in the run summary JSON only. Pin the
    contract so a future refactor doesn't quietly widen `.pipeline_status.json`
    without matching the docs.
    """
    from podcast_scraper.monitor import status as monitor_status

    class _FakeCfg:
        monitor = True

    monitor_status.maybe_update_pipeline_status(
        _FakeCfg(),
        str(tmp_path),
        stage="transcription",
        episode_idx=1,
        episode_total=3,
    )
    payload = monitor_status.read_pipeline_status(str(tmp_path))
    assert payload is not None
    # These are the fields `.pipeline_status.json` actually carries.
    for expected in ("pid", "stage", "started_at", "stage_started_at"):
        assert expected in payload
    # And these must NOT be there — they belong to the run summary JSON.
    for parallel_only_key in (
        "processing_overlap_ratio",
        "processing_thread_busy_ratio",
        "processing_thread_queue_idle_seconds",
        "inline_processed_episodes_count",
        "safety_net_processed_episodes_count",
        "handoff_latency_seconds_per_episode",
    ):
        assert parallel_only_key not in payload, (
            f".pipeline_status.json now carries {parallel_only_key!r} — update "
            "docs/guides/PIPELINE_AND_WORKFLOW.md → 'Parallelism observability' "
            "section which currently says only the run summary carries these."
        )
