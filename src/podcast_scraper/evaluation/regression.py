"""Regression detection and quality gate enforcement.

This module implements regression rules for detecting quality regressions
in experiments compared to baselines. It enforces quality gates based on
configurable thresholds.

This implements RFC-041 Phase 1: Regression Rules & Quality Gates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegressionRule:
    """A single regression rule with threshold and severity."""

    def __init__(
        self,
        metric_name: str,
        threshold: float,
        direction: str = "decrease",  # "decrease" means regression if metric decreases
        severity: str = "error",  # "error", "warning", "info"
    ):
        """Initialize a regression rule.

        Args:
            metric_name: Name of the metric to check
                (e.g., "rougeL_f1", "avg_latency_ms")
            threshold: Threshold value for the metric
            direction: "decrease" (regression if metric decreases) or
                "increase" (regression if metric increases)
            severity: Severity level ("error", "warning", "info")
        """
        self.metric_name = metric_name
        self.threshold = threshold
        self.direction = direction
        self.severity = severity

    def check(
        self, experiment_value: Optional[float], baseline_value: Optional[float]
    ) -> Optional[Dict[str, Any]]:
        """Check if this rule is violated.

        Args:
            experiment_value: Experiment metric value
            baseline_value: Baseline metric value

        Returns:
            Regression dict if rule is violated, None otherwise
        """
        if experiment_value is None or baseline_value is None:
            return None

        delta = experiment_value - baseline_value

        if self.direction == "decrease":
            # Regression if metric decreased below threshold
            if delta < -self.threshold:
                return {
                    "rule": self.metric_name,
                    "severity": self.severity,
                    "experiment_value": experiment_value,
                    "baseline_value": baseline_value,
                    "delta": delta,
                    "threshold": self.threshold,
                    "message": (
                        f"{self.metric_name} decreased by {abs(delta):.4f} "
                        f"(threshold: {self.threshold:.4f})"
                    ),
                }
        elif self.direction == "increase":
            # Regression if metric increased above threshold
            if delta > self.threshold:
                return {
                    "rule": self.metric_name,
                    "severity": self.severity,
                    "experiment_value": experiment_value,
                    "baseline_value": baseline_value,
                    "delta": delta,
                    "threshold": self.threshold,
                    "message": (
                        f"{self.metric_name} increased by {delta:.4f} "
                        f"(threshold: {self.threshold:.4f})"
                    ),
                }

        return None


class RegressionChecker:
    """Checks experiments against regression rules."""

    def __init__(self, rules: Optional[List[RegressionRule]] = None):
        """Initialize regression checker.

        Args:
            rules: Optional list of custom regression rules. If None, uses default rules.
        """
        self.rules = rules or self._default_rules()

    @staticmethod
    def _default_rules() -> List[RegressionRule]:
        """Get default regression rules for summarization.

        Returns:
            List of default regression rules
        """
        return [
            # Quality metrics - regress if they decrease
            RegressionRule("rougeL_f1", threshold=0.05, direction="decrease", severity="error"),
            RegressionRule("rouge1_f1", threshold=0.05, direction="decrease", severity="error"),
            RegressionRule("rouge2_f1", threshold=0.03, direction="decrease", severity="warning"),
            RegressionRule("bleu", threshold=0.05, direction="decrease", severity="warning"),
            RegressionRule(
                "embedding_cosine", threshold=0.05, direction="decrease", severity="warning"
            ),
            # Quality gates - regress if they increase
            RegressionRule(
                "boilerplate_leak_rate", threshold=0.0, direction="increase", severity="error"
            ),
            RegressionRule(
                "speaker_label_leak_rate",
                threshold=0.0,
                direction="increase",
                severity="error",
            ),
            RegressionRule(
                "truncation_rate", threshold=0.0, direction="increase", severity="error"
            ),
            # Performance - regress if latency increases significantly
            RegressionRule(
                "avg_latency_ms", threshold=1000.0, direction="increase", severity="warning"
            ),  # 1 second
            # Cost - warn if cost increases significantly
            RegressionRule(
                "total_cost_usd", threshold=0.10, direction="increase", severity="warning"
            ),  # $0.10
        ]

    @classmethod
    def from_config(cls, config_path: Path) -> RegressionChecker:
        """Load regression rules from YAML config file.

        Args:
            config_path: Path to regression rules YAML file

        Returns:
            RegressionChecker instance with loaded rules

        Example YAML format:
            rules:
              - metric: rougeL_f1
                threshold: 0.05
                direction: decrease
                severity: error
              - metric: avg_latency_ms
                threshold: 1000.0
                direction: increase
                severity: warning
        """
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rules = []

        for rule_config in config.get("rules", []):
            rule = RegressionRule(
                metric_name=rule_config["metric"],
                threshold=float(rule_config["threshold"]),
                direction=rule_config.get("direction", "decrease"),
                severity=rule_config.get("severity", "error"),
            )
            rules.append(rule)

        return cls(rules=rules)

    def check_comparison(self, comparison: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check a comparison result against regression rules.

        Args:
            comparison: Comparison dictionary from comparator.py

        Returns:
            List of regression violations (empty if none)
        """
        regressions = []
        deltas = comparison.get("deltas", {})

        for rule in self.rules:
            # Check if this rule's metric is in the deltas
            if rule.metric_name in deltas:
                delta = deltas[rule.metric_name]
                # We need baseline and experiment values to check the rule
                # For now, we'll use a simplified check based on delta
                # TODO: Load full metrics to get actual values
                violation = rule.check(
                    experiment_value=delta,  # This is a simplification
                    baseline_value=0.0,  # Baseline is 0 (delta = exp - baseline)
                )
                if violation:
                    regressions.append(violation)

        return regressions

    def check_metrics(
        self,
        experiment_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Check experiment metrics against baseline metrics using rules.

        Args:
            experiment_metrics: Experiment metrics dictionary
            baseline_metrics: Baseline metrics dictionary

        Returns:
            List of regression violations (empty if none)
        """
        regressions = []

        for rule in self.rules:
            # Extract metric value from experiment and baseline
            exp_value = self._extract_metric_value(experiment_metrics, rule.metric_name)
            baseline_value = self._extract_metric_value(baseline_metrics, rule.metric_name)

            violation = rule.check(exp_value, baseline_value)
            if violation:
                regressions.append(violation)

        return regressions

    def _extract_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract a metric value from metrics dictionary.

        Supports nested paths and vs_reference metrics:
        - "intrinsic.gates.boilerplate_leak_rate"
        - "vs_reference.silver_gpt4o_v1.rougeL_f1"
        - "rougeL_f1" (checks vs_reference if available)
        - "entity_set.precision" (checks vs_reference.{ref_id}.entity_set.precision
            for all refs)
        - "entity_set.per_label_f1.PERSON" (checks
            vs_reference.{ref_id}.entity_set.per_label_f1.PERSON)

        Args:
            metrics: Metrics dictionary
            metric_name: Name of metric to extract

        Returns:
            Metric value or None if not found
        """
        # Handle NER entity_set paths (e.g., "entity_set.precision",
        # "entity_set.per_label_f1.PERSON")
        if metric_name.startswith("entity_set."):
            vs_ref = metrics.get("vs_reference")
            if vs_ref and isinstance(vs_ref, dict):
                # Try each reference
                for ref_metrics in vs_ref.values():
                    if isinstance(ref_metrics, dict) and "entity_set" in ref_metrics:
                        # Build path: entity_set.{rest}
                        entity_set_path = metric_name.replace("entity_set.", "")
                        parts = ["entity_set"] + entity_set_path.split(".")
                        value: Any = ref_metrics
                        for part in parts:
                            if isinstance(value, dict):
                                value = value.get(part)
                            else:
                                break
                            if value is None:
                                break
                        if isinstance(value, (int, float)):
                            return float(value)

        # Handle simple metric names that might be in vs_reference
        if "." not in metric_name:
            # Check vs_reference first (common case for ROUGE, BLEU, etc.)
            vs_ref = metrics.get("vs_reference")
            if vs_ref and isinstance(vs_ref, dict):
                # Try each reference
                for ref_metrics in vs_ref.values():
                    if isinstance(ref_metrics, dict) and metric_name in ref_metrics:
                        value = ref_metrics[metric_name]
                        if isinstance(value, (int, float)):
                            return float(value)

        # Handle nested paths
        parts = metric_name.split(".")
        nested_value: Any = metrics

        for part in parts:
            if isinstance(nested_value, dict):
                nested_value = nested_value.get(part)
            else:
                return None

            if nested_value is None:
                return None

        if isinstance(nested_value, (int, float)):
            return float(nested_value)
        elif isinstance(nested_value, list):
            # For lists, return length (e.g., failed_episodes)
            return float(len(nested_value))

        return None

    def check_ner_invariants(
        self, metrics: Dict[str, Any], reference_id: Optional[str] = None
    ) -> List[str]:
        """Check NER-specific invariants (hard guardrails, not regressions).

        These are hard failures that should block before regression rules run:
        - Task type must be "ner_entities"
        - Reference must be present under vs_reference
        - Dataset must match between run and reference
        - For entity_set mode: scoring mode must be entity_set
        - Fingerprint mismatches (handled separately in scorer)

        Args:
            metrics: Metrics dictionary
            reference_id: Optional reference ID to check (if None, checks all references)

        Returns:
            List of invariant violation messages (empty if all pass)
        """
        violations = []

        # Check task type
        task_type = metrics.get("task")
        if task_type != "ner_entities":
            violations.append(
                f"NER invariant violation: task type is '{task_type}', expected 'ner_entities'"
            )
            return violations  # Early return if task type is wrong

        # Check vs_reference exists
        vs_reference = metrics.get("vs_reference")
        if not vs_reference or not isinstance(vs_reference, dict):
            violations.append("NER invariant violation: vs_reference is missing or invalid")
            return violations  # Early return if no vs_reference

        # Check specific reference if provided
        if reference_id:
            if reference_id not in vs_reference:
                violations.append(
                    f"NER invariant violation: reference '{reference_id}' not found in vs_reference"
                )
                return violations  # Early return if reference missing

            ref_metrics = vs_reference[reference_id]
            if not isinstance(ref_metrics, dict):
                violations.append(
                    f"NER invariant violation: reference '{reference_id}' metrics are invalid"
                )
                return violations

            # Check entity_set exists (required for NER gating)
            if "entity_set" not in ref_metrics:
                violations.append(
                    f"NER invariant violation: reference '{reference_id}' "
                    f"missing entity_set metrics. "
                    "Entity-set scoring is required for NER baselines. "
                    "Ensure scoring.mode includes 'entity_set' in experiment config."
                )

        return violations

    def check_gates(self, gates: Dict[str, Any]) -> List[str]:
        """Check quality gates for failures.

        Args:
            gates: Gates dictionary from metrics

        Returns:
            List of failed gate names
        """
        failed = []

        # Check each gate rate
        for gate_name in [
            "boilerplate_leak_rate",
            "speaker_label_leak_rate",
            "truncation_rate",
        ]:
            rate = gates.get(gate_name, 0.0)
            if rate > 0.0:
                failed.append(gate_name)

        # Check failed episodes
        failed_episodes = gates.get("failed_episodes", [])
        if failed_episodes:
            failed.append("failed_episodes")

        return failed

    def should_block_ci(self, regressions: List[Dict[str, Any]]) -> bool:
        """Determine if regressions should block CI.

        Args:
            regressions: List of regression violations

        Returns:
            True if CI should be blocked (any error-level regressions)
        """
        return any(r.get("severity") == "error" for r in regressions)
