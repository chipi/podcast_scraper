"""Pipeline cost/volume Counters (P2.9) — needs prometheus_client, so integration (not unit)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

pytest.importorskip("prometheus_client")

from podcast_scraper.server import pipeline_run_prometheus as prm  # noqa: E402


def _val(key: str) -> float:
    c = prm._PROM_STATE.get(key)
    return float(c._value.get()) if c is not None else 0.0


def test_observe_increments_cost_and_volume_counters() -> None:
    """cost / episodes / GI / KG Counters increment from a run.json metrics mapping (P2.9)."""
    prm._observe_metrics_mapping({"episodes_scraped_total": 1})  # ensure Counters exist
    before = {
        k: _val(k) for k in ("run_cost_usd", "run_episodes", "run_gi_artifacts", "run_kg_artifacts")
    }

    prm._observe_metrics_mapping(
        {
            "llm_gi_cost_usd": 2.0,
            "llm_kg_cost_usd": 1.0,
            "llm_transcription_cost_usd": 0.5,
            "episodes_scraped_total": 3,
            "gi_artifacts_generated": 3,
            "kg_artifacts_generated": 2,
        }
    )
    assert _val("run_cost_usd") == before["run_cost_usd"] + 3.5
    assert _val("run_episodes") == before["run_episodes"] + 3
    assert _val("run_gi_artifacts") == before["run_gi_artifacts"] + 3
    assert _val("run_kg_artifacts") == before["run_kg_artifacts"] + 2


def _jobs_counter_value(status: str, command_type: str) -> float:
    prm._ensure_prom_hist()
    ctr = prm._PROM_STATE.get("jobs_finished")
    if ctr is None:
        return 0.0
    return float(ctr.labels(status=status, command_type=command_type)._value.get())


def test_terminal_metrics_label_the_counter_by_command_type(monkeypatch, tmp_path) -> None:
    """The nightly and an operator-triggered enrichment must land on SEPARATE series.

    Sharing one series is what left the stalled-ingestion alert unable to tell "the nightly is
    dead" from "something, anything, succeeded recently" (#2119 follow-up).
    """
    monkeypatch.setenv("PODCAST_METRICS_ENABLED", "1")
    # No run.json in the corpus, so the call returns right after the counter increment — this
    # isolates the labelling from the run.json discovery path.
    monkeypatch.setattr(prm, "discover_run_json_paths_in_mtime_window", lambda *a, **k: [])

    before_nightly = _jobs_counter_value("succeeded", "full_incremental_pipeline")
    before_enrich = _jobs_counter_value("succeeded", "corpus_enrichment")

    prm.observe_pipeline_terminal_metrics(
        tmp_path, {"status": "succeeded", "command_type": "full_incremental_pipeline"}
    )

    assert _jobs_counter_value("succeeded", "full_incremental_pipeline") == before_nightly + 1
    # The enrichment series must NOT have moved — that separation is the entire point.
    assert _jobs_counter_value("succeeded", "corpus_enrichment") == before_enrich


def test_unknown_command_type_lands_on_other_not_its_own_series(monkeypatch, tmp_path) -> None:
    """Cardinality guard: an unrecognised value must not mint a new series."""
    monkeypatch.setenv("PODCAST_METRICS_ENABLED", "1")
    monkeypatch.setattr(prm, "discover_run_json_paths_in_mtime_window", lambda *a, **k: [])

    before = _jobs_counter_value("failed", "other")
    prm.observe_pipeline_terminal_metrics(
        tmp_path, {"status": "failed", "command_type": "adhoc-run-9e2f"}
    )
    assert _jobs_counter_value("failed", "other") == before + 1


def test_install_creates_every_job_series_at_zero_in_a_fresh_process() -> None:
    """A restarted API must export the series at 0 BEFORE the first job, not at its first value.

    Run in a subprocess: the registry here already holds series other tests incremented, which
    is exactly the state a freshly restarted API does not have.
    """
    code = (
        "from prometheus_client import generate_latest\n"
        "from podcast_scraper.server import pipeline_run_prometheus as prm\n"
        "assert prm.install_job_metrics() is True\n"
        "print(generate_latest().decode())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    for command_type in (*prm._KNOWN_COMMAND_TYPES, "other"):
        for status in ("succeeded", "failed", "cancelled", "stale"):
            line = (
                f'podcast_pipeline_jobs_finished_total{{command_type="{command_type}",'
                f'status="{status}"}} 0.0'
            )
            assert line in out, line
        assert (
            f'podcast_pipeline_run_episodes_processed_total{{command_type="{command_type}"}} 0.0'
            in out
        )
        assert (
            f'podcast_pipeline_run_wall_seconds_count{{command_type="{command_type}"}} 0.0' in out
        )


def test_terminal_metrics_observe_one_wall_sample_and_sum_processed_episodes(
    monkeypatch, tmp_path
) -> None:
    """One job over N per-feed run.json files is ONE wall observation, N files summed."""
    monkeypatch.setenv("PODCAST_METRICS_ENABLED", "1")
    prm._ensure_prom_hist()
    wall = prm._PROM_STATE["run_wall"].labels(command_type="full_incremental_pipeline")
    processed = prm._PROM_STATE["run_episodes_processed"].labels(
        command_type="full_incremental_pipeline"
    )
    count_before = wall._sum.get(), sum(b.get() for b in wall._buckets)
    processed_before = processed._value.get()
    hits_before = _val("run_json_hits")

    now = time.time()
    for i, (downloaded, transcribed) in enumerate([(2, 3), (0, 4), (1, 0)]):
        feed = tmp_path / "feeds" / f"f{i}"
        feed.mkdir(parents=True)
        run_json = feed / "run.json"
        run_json.write_text(
            json.dumps(
                {
                    "metrics": {
                        "transcripts_downloaded": downloaded,
                        "transcripts_transcribed": transcribed,
                        "episodes_scraped_total": 50,
                    }
                }
            ),
            encoding="utf-8",
        )
        os.utime(run_json, (now, now))

    started = datetime.fromtimestamp(now - 7200, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ended = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prm.observe_pipeline_terminal_metrics(
        tmp_path,
        {
            "status": "succeeded",
            "command_type": "full_incremental_pipeline",
            "started_at": started,
            "ended_at": ended,
        },
    )

    assert sum(b.get() for b in wall._buckets) == count_before[1] + 1
    assert wall._sum.get() == pytest.approx(count_before[0] + 7200, abs=2)
    assert processed._value.get() == processed_before + 10
    assert _val("run_json_hits") == hits_before + 3
