"""Tests for evaluation report generator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from podcast_scraper.evaluation.reporter import (
    format_delta,
    format_metric_value,
    generate_comparison_report,
    generate_metrics_report,
    print_report,
    save_report,
)


@pytest.mark.unit
class TestFormatMetricValue:
    """Test format_metric_value."""

    def test_none_returns_na(self):
        assert format_metric_value(None) == "N/A"

    def test_float_default(self):
        assert format_metric_value(0.1234) == "0.1234"

    def test_percentage(self):
        assert format_metric_value(0.5, "percentage") == "50.0%"

    def test_currency(self):
        assert format_metric_value(1.5, "currency") == "$1.5000"

    def test_duration(self):
        assert format_metric_value(100.7, "duration") == "101ms"

    def test_unknown_type_returns_str(self):
        assert format_metric_value(42, "unknown") == "42"


@pytest.mark.unit
class TestFormatDelta:
    """Test format_delta."""

    def test_none_returns_na(self):
        assert format_delta(None) == "N/A"

    def test_positive_prefix(self):
        assert format_delta(1.0, "float") == "+1.0000"

    def test_negative_prefix(self):
        assert format_delta(-0.5, "percentage") == "-50.0%"


@pytest.mark.unit
class TestGenerateMetricsReport:
    """Test generate_metrics_report."""

    def test_minimal_metrics(self):
        report = generate_metrics_report({"run_id": "r1", "dataset_id": "d1"})
        assert "# Experiment Metrics Report" in report
        assert "r1" in report
        assert "d1" in report

    def test_with_intrinsic_gates(self):
        metrics = {
            "run_id": "r1",
            "intrinsic": {
                "gates": {
                    "boilerplate_leak_rate": 0.0,
                    "speaker_label_leak_rate": 0.0,
                    "truncation_rate": 0.01,
                }
            },
        }
        report = generate_metrics_report(metrics)
        assert "Intrinsic Metrics" in report
        assert "Quality Gates" in report
        assert "0.0%" in report

    def test_with_vs_reference_and_error(self):
        metrics = {
            "run_id": "r1",
            "vs_reference": {"ref1": {"error": "missing baseline"}},
        }
        report = generate_metrics_report(metrics)
        assert "vs Reference" in report
        assert "missing baseline" in report


@pytest.mark.unit
class TestGenerateComparisonReport:
    """Test generate_comparison_report."""

    def test_minimal_comparison(self):
        report = generate_comparison_report(
            {"baseline_id": "b1", "experiment_run_id": "e1", "deltas": {}}
        )
        assert "Baseline Comparison Report" in report
        assert "b1" in report
        assert "No deltas computed" in report

    def test_with_cost_delta(self):
        report = generate_comparison_report(
            {"baseline_id": "b1", "deltas": {"cost_total_usd": -0.5}}
        )
        assert "Cost" in report
        assert "Cost decreased" in report

    def test_with_gate_regressions(self):
        report = generate_comparison_report(
            {"baseline_id": "b1", "deltas": {"gate_regressions": ["boilerplate_leak"]}}
        )
        assert "Quality Gate Regressions" in report
        assert "boilerplate_leak" in report


@pytest.mark.unit
class TestSaveReport:
    """Test save_report."""

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "report.md"
            save_report("# Hello", path)
            assert path.exists()
            assert path.read_text() == "# Hello"


@pytest.mark.unit
class TestPrintReport:
    """Test print_report (capture stdout)."""

    def test_print_report(self, capsys):
        print_report("# Test Report")
        out, _ = capsys.readouterr()
        assert "Test Report" in out
        assert "=" in out
