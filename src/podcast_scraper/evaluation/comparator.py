"""Comparator for computing deltas between runs.

This module computes deltas between an experiment run and a baseline.
Comparisons are separate from metrics to allow recomputation without re-running inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file.

    Args:
        metrics_path: Path to metrics.json

    Returns:
        Metrics dictionary
    """
    return dict(json.loads(metrics_path.read_text(encoding="utf-8")))  # type: ignore[no-any-return]


def compute_delta(
    experiment_value: Optional[float],
    baseline_value: Optional[float],
) -> Optional[float]:
    """Compute delta between experiment and baseline values.

    Args:
        experiment_value: Experiment metric value
        baseline_value: Baseline metric value

    Returns:
        Delta (experiment - baseline), or None if either value is None
    """
    if experiment_value is None or baseline_value is None:
        return None
    return experiment_value - baseline_value


def compare_vs_baseline(
    experiment_metrics_path: Path,
    baseline_metrics_path: Path,
    baseline_id: str,
    dataset_id: str,
) -> Dict[str, Any]:
    """Compute deltas vs baseline.

    Args:
        experiment_metrics_path: Path to experiment metrics.json
        baseline_metrics_path: Path to baseline metrics.json
        baseline_id: Baseline identifier
        dataset_id: Dataset identifier

    Returns:
        Comparison dictionary with deltas
    """
    # Load metrics
    exp_metrics = load_metrics(experiment_metrics_path)
    baseline_metrics = load_metrics(baseline_metrics_path)

    # Validate dataset_id matches
    if exp_metrics.get("dataset_id") != dataset_id:
        raise ValueError(
            f"Dataset mismatch: experiment uses dataset_id='{exp_metrics.get('dataset_id')}', "
            f"expected '{dataset_id}'"
        )
    if baseline_metrics.get("dataset_id") != dataset_id:
        raise ValueError(
            f"Dataset mismatch: baseline uses dataset_id='{baseline_metrics.get('dataset_id')}', "
            f"expected '{dataset_id}'"
        )

    deltas: Dict[str, Any] = {}

    # Intrinsic metric deltas
    exp_intrinsic = exp_metrics.get("intrinsic", {})
    baseline_intrinsic = baseline_metrics.get("intrinsic", {})

    # Cost deltas
    exp_cost = exp_intrinsic.get("cost", {})
    baseline_cost = baseline_intrinsic.get("cost", {})
    if (
        exp_cost.get("total_cost_usd") is not None
        and baseline_cost.get("total_cost_usd") is not None
    ):
        deltas["cost_total_usd"] = compute_delta(
            exp_cost.get("total_cost_usd"), baseline_cost.get("total_cost_usd")
        )

    # Performance deltas
    exp_perf = exp_intrinsic.get("performance", {})
    baseline_perf = baseline_intrinsic.get("performance", {})
    if (
        exp_perf.get("avg_latency_ms") is not None
        and baseline_perf.get("avg_latency_ms") is not None
    ):
        deltas["avg_latency_ms"] = compute_delta(
            exp_perf.get("avg_latency_ms"), baseline_perf.get("avg_latency_ms")
        )

    # Gate regressions (hard fail if any gate fails)
    exp_gates = exp_intrinsic.get("gates", {})
    baseline_gates = baseline_intrinsic.get("gates", {})
    gate_regressions = []
    for gate_name in ["boilerplate_leak_rate", "speaker_leak_rate", "truncation_rate"]:
        exp_rate = exp_gates.get(gate_name, 0.0)
        baseline_rate = baseline_gates.get(gate_name, 0.0)
        if exp_rate > baseline_rate:
            gate_regressions.append(gate_name)
    deltas["gate_regressions"] = gate_regressions

    # vs_reference deltas (if both have same references)
    exp_vs_ref = exp_metrics.get("vs_reference", {})
    baseline_vs_ref = baseline_metrics.get("vs_reference", {})
    if exp_vs_ref and baseline_vs_ref:
        for ref_id in exp_vs_ref.keys():
            if ref_id in baseline_vs_ref:
                exp_rougeL = exp_vs_ref[ref_id].get("rougeL_f1")
                baseline_rougeL = baseline_vs_ref[ref_id].get("rougeL_f1")
                if exp_rougeL is not None and baseline_rougeL is not None:
                    deltas[f"rougeL_f1_vs_{ref_id}"] = compute_delta(exp_rougeL, baseline_rougeL)

    return {
        "baseline_id": baseline_id,
        "dataset_id": dataset_id,
        "experiment_run_id": exp_metrics.get("run_id"),
        "deltas": deltas,
    }
