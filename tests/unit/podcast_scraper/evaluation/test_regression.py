"""Unit tests for podcast_scraper.evaluation.regression module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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

    def test_should_block_ci_false_when_empty(self):
        assert RegressionChecker().should_block_ci([]) is False


@pytest.mark.unit
class TestRegressionCheckerFromConfig:
    def test_from_config_builds_rules(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "reg.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "rules": [
                        {"metric": "custom_m", "threshold": 0.1, "direction": "increase"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        checker = RegressionChecker.from_config(cfg_path)
        assert len(checker.rules) == 1
        assert checker.rules[0].metric_name == "custom_m"
        assert checker.rules[0].threshold == pytest.approx(0.1)
        assert checker.rules[0].direction == "increase"


@pytest.mark.unit
class TestRegressionCheckerCheckComparison:
    def test_uses_delta_dict_when_metrics_missing(self) -> None:
        rule = RegressionRule("cost_total_usd", threshold=0.01, direction="increase")
        checker = RegressionChecker(rules=[rule])
        comparison = {
            "deltas": {"cost_total_usd": 0.5},
            "experiment_metrics": None,
            "baseline_metrics": None,
        }
        out = checker.check_comparison(comparison)
        assert len(out) == 1
        assert out[0]["rule"] == "cost_total_usd"

    def test_prefers_metrics_over_deltas_when_present(self) -> None:
        rule = RegressionRule("rougeL_f1", threshold=0.01, direction="decrease")
        checker = RegressionChecker(rules=[rule])
        comparison = {
            "deltas": {"rougeL_f1": 0.0},
            "experiment_metrics": {"vs_reference": {"r": {"rougeL_f1": 0.2}}},
            "baseline_metrics": {"vs_reference": {"r": {"rougeL_f1": 0.5}}},
        }
        out = checker.check_comparison(comparison)
        assert len(out) == 1
        assert out[0]["delta"] == pytest.approx(-0.3)


@pytest.mark.unit
class TestRegressionCheckerExtractMetric:
    def test_entity_set_precision_from_vs_reference(self) -> None:
        checker = RegressionChecker()
        metrics = {
            "vs_reference": {
                "silver": {
                    "entity_set": {"precision": 0.88, "recall": 0.5},
                }
            }
        }
        assert checker._extract_metric_value(metrics, "entity_set.precision") == pytest.approx(0.88)

    def test_nested_intrinsic_path(self) -> None:
        checker = RegressionChecker()
        metrics = {"intrinsic": {"gates": {"boilerplate_leak_rate": 0.03}}}
        assert checker._extract_metric_value(
            metrics, "intrinsic.gates.boilerplate_leak_rate"
        ) == pytest.approx(0.03)


@pytest.mark.unit
class TestRegressionCheckerNerInvariants:
    def test_wrong_task_type(self) -> None:
        checker = RegressionChecker()
        v = checker.check_ner_invariants({"task": "summarization"})
        assert len(v) == 1
        assert "ner_entities" in v[0]

    def test_missing_vs_reference(self) -> None:
        checker = RegressionChecker()
        v = checker.check_ner_invariants({"task": "ner_entities"})
        assert any("vs_reference" in msg for msg in v)

    def test_reference_id_not_found(self) -> None:
        checker = RegressionChecker()
        v = checker.check_ner_invariants(
            {"task": "ner_entities", "vs_reference": {"other": {}}},
            reference_id="missing",
        )
        assert any("missing" in msg for msg in v)
