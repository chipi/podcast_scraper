"""Unit tests for podcast_scraper.evaluation.regression module."""

from __future__ import annotations

import pytest

from podcast_scraper.evaluation.regression import (
    RegressionChecker,
    RegressionRule,
)


@pytest.mark.unit
class TestRegressionRule:
    """Tests for RegressionRule."""

    def test_check_none_values_returns_none(self):
        """None experiment or baseline value returns None."""
        rule = RegressionRule("rougeL_f1", threshold=0.05, direction="decrease")
        assert rule.check(None, 0.5) is None
        assert rule.check(0.5, None) is None

    def test_decrease_direction_regression_below_threshold(self):
        """Decrease direction: regression when metric drops below threshold."""
        rule = RegressionRule("rougeL_f1", threshold=0.05, direction="decrease")
        # baseline 0.5, experiment 0.44 -> delta -0.06 < -0.05 -> violation
        out = rule.check(experiment_value=0.44, baseline_value=0.5)
        assert out is not None
        assert out["rule"] == "rougeL_f1"
        assert out["severity"] == "error"
        assert out["delta"] == pytest.approx(-0.06)

    def test_decrease_direction_no_regression_when_above_threshold(self):
        """Decrease direction: no violation when drop is within threshold."""
        rule = RegressionRule("rougeL_f1", threshold=0.05, direction="decrease")
        # baseline 0.5, experiment 0.46 -> delta -0.04 > -0.05 -> no violation
        assert rule.check(experiment_value=0.46, baseline_value=0.5) is None

    def test_increase_direction_regression_above_threshold(self):
        """Increase direction: regression when metric increases above threshold."""
        rule = RegressionRule(
            "boilerplate_leak_rate", threshold=0.0, direction="increase", severity="error"
        )
        out = rule.check(experiment_value=0.1, baseline_value=0.0)
        assert out is not None
        assert out["delta"] == pytest.approx(0.1)


@pytest.mark.unit
class TestRegressionChecker:
    """Tests for RegressionChecker."""

    def test_check_metrics_empty_when_no_violations(self):
        """check_metrics returns empty list when no rules violated."""
        checker = RegressionChecker()
        exp = {"vs_reference": {"ref1": {"rougeL_f1": 0.5}}}
        base = {"vs_reference": {"ref1": {"rougeL_f1": 0.5}}}
        regressions = checker.check_metrics(exp, base)
        assert regressions == []

    def test_check_metrics_finds_violation(self):
        """check_metrics returns violations when metric regresses."""
        checker = RegressionChecker()
        exp = {"vs_reference": {"ref1": {"rougeL_f1": 0.40}}}
        base = {"vs_reference": {"ref1": {"rougeL_f1": 0.50}}}
        regressions = checker.check_metrics(exp, base)
        assert len(regressions) >= 1
        assert any(r["rule"] == "rougeL_f1" for r in regressions)

    def test_check_gates_returns_failed_gate_names(self):
        """check_gates returns list of failed gate names."""
        checker = RegressionChecker()
        gates = {"boilerplate_leak_rate": 0.1, "speaker_label_leak_rate": 0.0}
        failed = checker.check_gates(gates)
        assert "boilerplate_leak_rate" in failed

    def test_check_gates_failed_episodes_included(self):
        """check_gates includes failed_episodes when non-empty."""
        checker = RegressionChecker()
        gates = {"failed_episodes": ["e1", "e2"]}
        failed = checker.check_gates(gates)
        assert "failed_episodes" in failed

    def test_should_block_ci_true_when_error_severity(self):
        """should_block_ci returns True when any regression has severity error."""
        checker = RegressionChecker()
        regressions = [{"severity": "error"}, {"severity": "warning"}]
        assert checker.should_block_ci(regressions) is True

    def test_should_block_ci_false_when_only_warnings(self):
        """should_block_ci returns False when only warnings."""
        checker = RegressionChecker()
        regressions = [{"severity": "warning"}]
        assert checker.should_block_ci(regressions) is False
