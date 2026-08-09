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
