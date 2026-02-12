"""Tests for evaluation history and comparison."""

from __future__ import annotations

import json

import pytest

from podcast_scraper.evaluation.history import (
    compare_experiments,
    find_all_baselines,
    find_all_runs,
    generate_history_report,
)


@pytest.mark.unit
class TestFindAllRuns:
    """Test find_all_runs."""

    def test_empty_when_dir_missing(self, tmp_path):
        """Returns empty list when base_dir does not exist."""
        out = find_all_runs(base_dir=tmp_path / "nonexistent")
        assert out == []

    def test_empty_when_dir_empty(self, tmp_path):
        """Returns empty list when base_dir has no subdirs."""
        out = find_all_runs(base_dir=tmp_path)
        assert out == []

    def test_finds_run_dirs(self, tmp_path):
        """Finds run directories and loads metadata."""
        (tmp_path / "run_001").mkdir()
        (tmp_path / "run_001" / "baseline.json").write_text(
            json.dumps({"created_at": "2025-01-01T00:00:00Z", "dataset_id": "d1"}),
            encoding="utf-8",
        )
        (tmp_path / "run_002").mkdir()
        out = find_all_runs(base_dir=tmp_path)
        assert len(out) == 2
        run_ids = {r["run_id"] for r in out}
        assert "run_001" in run_ids
        assert "run_002" in run_ids
        r1 = next(r for r in out if r["run_id"] == "run_001")
        assert r1.get("metadata", {}).get("dataset_id") == "d1"


@pytest.mark.unit
class TestFindAllBaselines:
    """Test find_all_baselines."""

    def test_empty_when_dir_missing(self, tmp_path):
        """Returns empty list when base_dir does not exist."""
        out = find_all_baselines(base_dir=tmp_path / "nonexistent")
        assert out == []

    def test_finds_baseline_dirs(self, tmp_path):
        """Finds baseline directories."""
        (tmp_path / "baseline_001").mkdir()
        (tmp_path / "baseline_001" / "baseline.json").write_text(
            json.dumps({"created_at": "2025-01-01"}), encoding="utf-8"
        )
        out = find_all_baselines(base_dir=tmp_path)
        assert len(out) == 1
        assert out[0]["baseline_id"] == "baseline_001"


@pytest.mark.unit
class TestCompareExperiments:
    """Test compare_experiments."""

    def test_raises_when_metrics_missing(self, tmp_path):
        """Raises FileNotFoundError when metrics.json missing."""
        (tmp_path / "r1").mkdir()
        (tmp_path / "r2").mkdir()
        with pytest.raises(FileNotFoundError, match="Metrics not found"):
            compare_experiments(tmp_path / "r1", tmp_path / "r2")

    def test_returns_deltas_when_both_have_metrics(self, tmp_path):
        """Returns comparison dict with deltas when both have metrics."""
        (tmp_path / "r1").mkdir()
        (tmp_path / "r2").mkdir()
        (tmp_path / "r1" / "metrics.json").write_text(
            json.dumps(
                {
                    "run_id": "r1",
                    "dataset_id": "d1",
                    "intrinsic": {
                        "cost": {"total_cost_usd": 1.0},
                        "performance": {"avg_latency_ms": 100.0},
                        "gates": {
                            "boilerplate_leak_rate": 0.0,
                            "speaker_label_leak_rate": 0.0,
                            "truncation_rate": 0.01,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "r2" / "metrics.json").write_text(
            json.dumps(
                {
                    "run_id": "r2",
                    "dataset_id": "d1",
                    "intrinsic": {
                        "cost": {"total_cost_usd": 0.5},
                        "performance": {"avg_latency_ms": 80.0},
                        "gates": {
                            "boilerplate_leak_rate": 0.0,
                            "speaker_label_leak_rate": 0.0,
                            "truncation_rate": 0.01,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        result = compare_experiments(tmp_path / "r1", tmp_path / "r2", dataset_id="d1")
        assert "deltas" in result
        # compute_delta(run1, run2) = run1 - run2
        assert result["deltas"]["cost_total_usd"] == 0.5
        assert result["deltas"]["avg_latency_ms"] == 20.0
        assert result["run1_id"] == "r1"
        assert result["run2_id"] == "r2"


@pytest.mark.unit
class TestGenerateHistoryReport:
    """Test generate_history_report."""

    def test_empty_runs(self):
        """Returns report with no runs message."""
        report = generate_history_report([], metric_name="rougeL_f1")
        assert "Historical Trend Report" in report
        assert "No runs found" in report

    def test_with_runs_no_metrics_files(self, tmp_path):
        """Report includes run IDs when runs have no metrics.json."""
        runs = [
            {"run_id": "r1", "created_at": "2025-01-01", "path": tmp_path / "r1"},
        ]
        (tmp_path / "r1").mkdir()
        report = generate_history_report(runs, metric_name="rougeL_f1")
        assert "r1" in report
        assert "Trend Data" in report
