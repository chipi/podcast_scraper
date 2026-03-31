"""Integration tests for tools/run_compare data loading (RFC-047, Issue #373)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tools.run_compare.data import (
    artifact_status,
    compute_per_episode_rouge_rows,
    discover_runs,
    extract_aggregate_rouge,
    extract_kpis,
    extract_rouge_l_f1,
    index_predictions,
    load_metrics,
    load_predictions_jsonl,
    merge_run_summary,
    pick_shared_reference_id,
    predictions_to_chart_rows,
    rouge_comparable_episode_ids,
)


def _write_minimal_run(run_dir: Path, run_id: str, failed_episodes: list) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset_id": "ds_test",
        "run_id": run_id,
        "episode_count": 2,
        "intrinsic": {
            "gates": {
                "boilerplate_leak_rate": 0.0,
                "speaker_label_leak_rate": 0.0,
                "truncation_rate": 0.1,
                "failed_episodes": failed_episodes,
                "episode_gate_failures": {},
            },
            "length": {"avg_tokens": 120.0},
            "performance": {"avg_latency_ms": 8000.0},
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    preds = [
        {
            "episode_id": "p01_e01",
            "output": {"summary_final": "hello world"},
            "metadata": {
                "processing_time_seconds": 10.0,
                "output_length_chars": 400,
            },
        },
        {
            "episode_id": "p02_e01",
            "error": "inference failed",
        },
    ]
    lines = "\n".join(json.dumps(p) for p in preds)
    (run_dir / "predictions.jsonl").write_text(lines + "\n", encoding="utf-8")
    (run_dir / "fingerprint.json").write_text("{}", encoding="utf-8")


@pytest.mark.integration
def test_discover_runs_finds_baselines(tmp_path: Path) -> None:
    b1 = tmp_path / "eval" / "baselines" / "b_one"
    _write_minimal_run(b1, "b_one", [])
    found = discover_runs(tmp_path / "eval")
    assert len(found) == 1
    assert found[0].run_id == "b_one"
    assert found[0].category == "baseline"


@pytest.mark.integration
def test_extract_kpis_and_artifact_status(tmp_path: Path) -> None:
    rdir = tmp_path / "eval" / "runs" / "r1"
    _write_minimal_run(rdir, "r1", ["p02_e01"])
    st = artifact_status(rdir)
    assert st["metrics.json"] is True
    assert st["predictions.jsonl"] is True
    assert st["diagnostics.jsonl"] is False
    m = load_metrics(rdir)
    k = extract_kpis(m)
    assert k["episode_count"] == 2
    assert k["failed_count"] == 1
    assert k["truncation_rate"] == 0.1


@pytest.mark.integration
def test_load_predictions_jsonl_and_chart_rows(tmp_path: Path) -> None:
    rdir = tmp_path / "eval" / "runs" / "r2"
    _write_minimal_run(rdir, "r2", [])
    preds = load_predictions_jsonl(rdir / "predictions.jsonl")
    assert len(preds) == 2
    # ROUGE-comparable only: p02_e01 has inference error; excluded from charts.
    rows = predictions_to_chart_rows("run", preds, failed_ids=["p02_e01"])
    assert len(rows) == 1
    assert rows[0]["episode_id"] == "p01_e01"
    assert rows[0]["failed"] is False


@pytest.mark.integration
def test_extract_aggregate_rouge() -> None:
    m = {
        "vs_reference": {
            "r1": {"rouge1_f1": 0.1, "rouge2_f1": 0.2, "rougeL_f1": 0.3},
        }
    }
    hit = extract_aggregate_rouge(m)
    assert hit is not None
    rid, blob = hit
    assert rid == "r1"
    assert blob["rougeL_f1"] == pytest.approx(0.3)


@pytest.mark.integration
def test_pick_shared_reference_id_order() -> None:
    loaded = {
        "A": {"metrics": {"vs_reference": {"x": {"rougeL_f1": 0.1}}}},
        "B": {"metrics": {"vs_reference": {"y": {"rougeL_f1": 0.2}}}},
    }
    r1, _ = pick_shared_reference_id(loaded, ordered_labels=["A", "B"])
    assert r1 == "x"
    r2, _ = pick_shared_reference_id(loaded, ordered_labels=["B", "A"])
    assert r2 == "y"


@pytest.mark.integration
def test_extract_rouge_l_f1_from_metrics() -> None:
    m = {
        "vs_reference": {
            "silver_x": {"rougeL_f1": 0.33, "rouge1_f1": 0.4},
        }
    }
    assert extract_rouge_l_f1(m) == pytest.approx(0.33)
    assert extract_rouge_l_f1({}) is None
    assert extract_rouge_l_f1({"vs_reference": {"bad": {"error": "x"}}}) is None


@pytest.mark.integration
def test_compute_per_episode_rouge_rows() -> None:
    ref_preds = [
        {"episode_id": "e1", "output": {"summary_final": "the cat sat on the mat"}},
    ]
    run_preds = [
        {"episode_id": "e1", "output": {"summary_final": "the cat sat on the mat"}},
    ]
    ref_by = index_predictions(ref_preds)
    rows = compute_per_episode_rouge_rows("run_a", run_preds, [], ref_by)
    assert len(rows) == 1
    assert rows[0]["rougeL_f1"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.integration
def test_rouge_comparable_episode_ids() -> None:
    preds: List[Dict[str, Any]] = [
        {"episode_id": "a", "output": {"summary_final": "ok"}},
        {"episode_id": "b", "error": "bad"},
        {"episode_id": "c", "output": {"summary_final": ""}},
    ]
    s = rouge_comparable_episode_ids(preds, failed_ids=["c"])
    assert s == {"a"}


@pytest.mark.integration
def test_merge_run_summary_overrides(tmp_path: Path) -> None:
    rdir = tmp_path / "eval" / "runs" / "r3"
    _write_minimal_run(rdir, "r3", [])
    summary = {"avg_output_tokens": 999.0, "avg_latency_s": 42.0}
    (rdir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    m0 = load_metrics(rdir)
    m1 = merge_run_summary(m0, rdir)
    k = extract_kpis(m1)
    assert k["avg_output_tokens"] == 999.0
    assert k["avg_latency_s"] == pytest.approx(42.0)


@pytest.mark.integration
def test_malformed_jsonl_line_skipped(tmp_path: Path) -> None:
    p = tmp_path / "pred.jsonl"
    p.write_text('{"a": 1}\nnot-json\n{"b": 2}\n', encoding="utf-8")
    rows = load_predictions_jsonl(p)
    assert len(rows) == 2
