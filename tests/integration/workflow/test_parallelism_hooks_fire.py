"""Integration guard: the #1180 metric hooks actually fire under a real pipeline path.

The unit tests in ``tests/unit/podcast_scraper/test_metrics_parallelism.py``
cover the interval math and the ``Metrics`` recording contract in isolation.
They do not prove that the transcription and processing thread entry points
actually CALL those record methods — a refactor that drops
``_stamp_tx_active`` or ``_time_processing_job`` would leave the unit tests
passing while the run summary silently reports None ratios.

This module drives the real ``process_transcription_jobs_concurrent`` and
``process_processing_jobs_concurrent`` stage functions with a mocked heavy
inner (Whisper / LLM) so we can assert the metric fields get populated
end-to-end without a real ML dependency. Runs in <1 s.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config, models
from podcast_scraper.workflow import metrics
from podcast_scraper.workflow.stages import processing, transcription
from podcast_scraper.workflow.types import (
    ProcessingJob,
    ProcessingResources,
    TranscriptionResources,
)

pytestmark = pytest.mark.integration


def _minimal_cfg() -> config.Config:
    # rss_urls omitted — this test drives the stages directly with a mocked
    # transcription queue, so the RSS feed itself is never fetched.
    return config.Config(
        transcription_provider="whisper",
        transcribe_missing=True,
        transcription_parallelism=1,
        processing_parallelism=1,
        generate_metadata=True,
    )


def _fake_transcribe_media_to_text(job, *_args, **_kwargs) -> tuple[bool, str, int]:
    """Stand-in for the real transcription call. Sleeps briefly so
    _process_single_job's monotonic-clock bracket has a measurable interval.
    """
    time.sleep(0.005)
    return True, f"/tmp/fake_ep{job.idx}.txt", 0


def _fake_generate_metadata(*_args, **_kwargs) -> None:
    """Stand-in for the metadata + summary + GI + KG per-episode chain.
    Sleep so the processing thread's active interval has measurable width.
    """
    time.sleep(0.005)


def _make_episode(idx: int) -> Any:
    """Build the smallest object the stages accept as an Episode."""
    ep = Mock(spec=models.Episode)
    ep.idx = idx
    ep.item = Mock()
    return ep


# ---------- Transcription-thread hook -------------------------------------


def test_transcription_stage_records_active_intervals() -> None:
    """After running the transcription stage on one queued job, the metrics
    should carry at least one recorded ``transcription_thread_intervals``
    entry. If a refactor drops ``_stamp_tx_active``, this fails.
    """
    cfg = _minimal_cfg()
    pm = metrics.Metrics()

    tx_jobs: queue.Queue = queue.Queue()
    # Use a bare tuple that mirrors TranscriptionJob's shape enough for the
    # sequential path — the stage only touches .idx.
    tx_job = Mock()
    tx_job.idx = 1
    tx_jobs.put(tx_job)

    tx_resources = TranscriptionResources(
        transcription_provider=None,
        temp_dir=None,
        transcription_jobs=tx_jobs,
        transcription_jobs_lock=None,
        saved_counter_lock=None,
    )
    proc_resources = ProcessingResources(
        processing_jobs=[],
        processing_jobs_lock=None,
        processing_complete_event=None,
    )
    downloads_done = threading.Event()
    downloads_done.set()

    feed = models.RssFeed(title="T", items=[], base_url="https://example.com", authors=[])

    with patch.object(transcription, "transcribe_media_to_text", _fake_transcribe_media_to_text):
        transcription.process_transcription_jobs_concurrent(
            transcription_resources=tx_resources,
            download_args=[(_make_episode(1),) + (None,) * 8],
            episodes=[_make_episode(1)],
            feed=feed,
            cfg=cfg,
            effective_output_dir="/tmp/fake",
            run_suffix=None,
            feed_metadata=Mock(),
            host_detection_result=Mock(),
            processing_resources=proc_resources,
            pipeline_metrics=pm,
            summary_provider=None,
            downloads_complete_event=downloads_done,
            saved_counter=[0],
        )

    assert pm.transcription_thread_intervals, (
        "process_transcription_jobs_concurrent did not populate "
        "transcription_thread_intervals — the _stamp_tx_active hook is not "
        "firing. Check stages/transcription.py::_process_single_job."
    )
    # And the interval must be real (start < end).
    for start, end in pm.transcription_thread_intervals:
        assert end > start, f"degenerate interval: ({start}, {end})"


# ---------- Processing-thread hook + inline counter -----------------------


def test_processing_stage_records_active_interval_and_inline_count() -> None:
    """Running the processing stage on a queued ProcessingJob must:
    - populate processing_thread_intervals (proves _time_processing_job fires),
    - bump inline_processed_episodes_count (proves the counter wiring works),
    - populate handoff_latency_seconds_per_episode when queued_at is stamped.
    """
    cfg = _minimal_cfg()
    pm = metrics.Metrics()

    episode = _make_episode(1)
    queued_at = time.monotonic()
    job = ProcessingJob(
        episode=episode,
        transcript_path="/tmp/fake_ep1.txt",
        transcript_source="whisper_transcription",
        detected_names=None,
        whisper_model="tiny.en",
        queued_at=queued_at,
    )
    proc_resources = ProcessingResources(
        processing_jobs=[job],
        processing_jobs_lock=None,
        processing_complete_event=None,
    )
    transcription_done = threading.Event()
    transcription_done.set()

    feed = models.RssFeed(title="T", items=[], base_url="https://example.com", authors=[])

    # Stub the transcript-existence wait AND the heavy metadata/LLM call so we
    # exercise only the framing code the hooks live in.
    with patch.object(
        processing, "metadata_stage", Mock(call_generate_metadata=_fake_generate_metadata)
    ):
        # _wait_for_transcript_file is a closure — patch via os.path.exists
        # so the "transcript missing" check passes without touching disk.
        with patch("os.path.exists", return_value=True):
            processing.process_processing_jobs_concurrent(
                processing_resources=proc_resources,
                feed=feed,
                cfg=cfg,
                effective_output_dir="/tmp/fake",
                run_suffix=None,
                feed_metadata=Mock(),
                host_detection_result=Mock(),
                pipeline_metrics=pm,
                summary_provider=None,
                transcription_complete_event=transcription_done,
                should_serialize_mps=False,
            )

    assert pm.processing_thread_intervals, (
        "process_processing_jobs_concurrent did not populate "
        "processing_thread_intervals — the _time_processing_job hook is not "
        "firing. Check stages/processing.py."
    )
    assert pm.inline_processed_episodes_count == 1, (
        f"expected inline_processed_episodes_count == 1, got "
        f"{pm.inline_processed_episodes_count} — the counter wiring in "
        f"_time_processing_job may have regressed."
    )
    assert pm.handoff_latency_seconds_per_episode, (
        "handoff_latency_seconds_per_episode is empty — either queued_at was "
        "not stamped on ProcessingJob or the recorder never ran."
    )
    # Handoff latency must be positive-ish (test scheduling can make it near-zero).
    assert pm.handoff_latency_seconds_per_episode[0] >= 0


# ---------- End-to-end wiring: finalize computes ratios ------------------


def test_finalize_produces_populated_ratios_after_stage_hooks_fire() -> None:
    """After BOTH stages ran hooks, finalize_parallelism_snapshot yields
    non-None ratios in the export dict. This is the "wiring works
    end-to-end" guard.
    """
    pm = metrics.Metrics()

    # Simulate two threads running in overlap. We don't need to invoke the
    # stages here — we just need the intervals in place, then finalize +
    # export must produce numbers.
    t0 = time.monotonic()
    pm.record_transcription_thread_active(t0, t0 + 1.0)
    pm.record_processing_thread_active(t0 + 0.5, t0 + 1.5)
    pm.record_inline_processed_episode()
    pm.finalize_parallelism_snapshot(pipeline_wall_seconds=1.5)

    d = pm.finish()

    assert d["processing_overlap_ratio"] is not None
    assert 0.0 < d["processing_overlap_ratio"] <= 1.0
    assert d["processing_thread_busy_ratio"] is not None
    assert 0.0 < d["processing_thread_busy_ratio"] <= 1.0
    assert d["inline_processed_episodes_count"] == 1
