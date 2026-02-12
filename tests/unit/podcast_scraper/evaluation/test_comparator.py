"""Tests for evaluation comparator."""

from __future__ import annotations

import json

import pytest

from podcast_scraper.evaluation.comparator import (
    compare_vs_baseline,
    compute_delta,
    load_metrics,
)


@pytest.mark.unit
class TestLoadMetrics:
    def test_load_metrics_from_file(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"run_id": "r1", "dataset_id": "d1"}), encoding="utf-8")
        out = load_metrics(path)
        assert out["run_id"] == "r1"
        assert out["dataset_id"] == "d1"


@pytest.mark.unit
class TestComputeDelta:
    def test_returns_none_when_either_none(self):
        assert compute_delta(None, 1.0) is None
        assert compute_delta(1.0, None) is None

    def test_returns_experiment_minus_baseline(self):
        assert compute_delta(2.0, 1.0) == 1.0
        assert compute_delta(0.5, 1.0) == -0.5


@pytest.mark.unit
class TestCompareVsBaseline:
    def test_raises_on_dataset_mismatch_experiment(self, tmp_path):
        (tmp_path / "exp.json").write_text(json.dumps({"dataset_id": "other"}), encoding="utf-8")
        (tmp_path / "base.json").write_text(json.dumps({"dataset_id": "d1"}), encoding="utf-8")
        with pytest.raises(ValueError, match="Dataset mismatch"):
            compare_vs_baseline(
                tmp_path / "exp.json",
                tmp_path / "base.json",
                baseline_id="b1",
                dataset_id="d1",
            )

    def test_returns_comparison_with_deltas(self, tmp_path):
        (tmp_path / "exp.json").write_text(
            json.dumps(
                {
                    "dataset_id": "d1",
                    "run_id": "exp1",
                    "intrinsic": {
                        "cost": {"total_cost_usd": 2.0},
                        "performance": {"avg_latency_ms": 150.0},
                        "gates": {},
                    },
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "base.json").write_text(
            json.dumps(
                {
                    "dataset_id": "d1",
                    "run_id": "base1",
                    "intrinsic": {
                        "cost": {"total_cost_usd": 1.0},
                        "performance": {"avg_latency_ms": 100.0},
                        "gates": {},
                    },
                }
            ),
            encoding="utf-8",
        )
        result = compare_vs_baseline(
            tmp_path / "exp.json",
            tmp_path / "base.json",
            baseline_id="b1",
            dataset_id="d1",
        )
        assert result["baseline_id"] == "b1"
        assert result["deltas"]["cost_total_usd"] == 1.0
        assert result["deltas"]["avg_latency_ms"] == 50.0
