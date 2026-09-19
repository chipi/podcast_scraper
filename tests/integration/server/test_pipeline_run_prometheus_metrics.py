"""Pipeline cost/volume Counters (P2.9) — needs prometheus_client, so integration (not unit)."""

from __future__ import annotations

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
