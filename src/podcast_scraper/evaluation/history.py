"""Historical tracking and comparison tools for experiment runs.

This module provides tools for:
- Tracking experiment runs over time
- Comparing any two experiments
- Generating historical trend reports

This implements RFC-015 Phase 3: Storage & Comparison.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def find_all_runs(base_dir: Path = Path("data/eval/runs")) -> List[Dict[str, Any]]:
    """Find all experiment runs in the runs directory.

    Args:
        base_dir: Base directory containing runs (default: data/eval/runs)

    Returns:
        List of run metadata dictionaries, sorted by creation time (newest first)
    """
    runs: List[Dict[str, Any]] = []

    if not base_dir.exists():
        return runs

    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        metrics_path = run_dir / "metrics.json"
        baseline_path = run_dir / "baseline.json"

        # Try to load metadata
        metadata = {}
        if baseline_path.exists():
            try:
                metadata = json.loads(baseline_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load baseline.json for {run_id}: {e}")

        # Try to load metrics
        metrics = None
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load metrics.json for {run_id}: {e}")

        # Extract creation time
        created_at = metadata.get("created_at") or metadata.get("promoted_at")
        if not created_at and metrics:
            created_at = metrics.get("run_id")  # Fallback to run_id if it's a timestamp

        runs.append(
            {
                "run_id": run_id,
                "path": run_dir,
                "created_at": created_at,
                "dataset_id": metrics.get("dataset_id") if metrics else metadata.get("dataset_id"),
                "has_metrics": metrics is not None,
                "metadata": metadata,
            }
        )

    # Sort by creation time (newest first)
    runs.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)

    return runs


def find_all_baselines(base_dir: Path = Path("data/eval/baselines")) -> List[Dict[str, Any]]:
    """Find all baselines in the baselines directory.

    Args:
        base_dir: Base directory containing baselines (default: data/eval/baselines)

    Returns:
        List of baseline metadata dictionaries, sorted by creation time (newest first)
    """
    baselines: List[Dict[str, Any]] = []

    if not base_dir.exists():
        return baselines

    for baseline_dir in base_dir.iterdir():
        if not baseline_dir.is_dir():
            continue

        baseline_id = baseline_dir.name
        baseline_json_path = baseline_dir / "baseline.json"
        metrics_path = baseline_dir / "metrics.json"

        # Try to load metadata
        metadata = {}
        if baseline_json_path.exists():
            try:
                metadata = json.loads(baseline_json_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load baseline.json for {baseline_id}: {e}")

        # Try to load metrics
        metrics = None
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load metrics.json for {baseline_id}: {e}")

        baselines.append(
            {
                "baseline_id": baseline_id,
                "path": baseline_dir,
                "created_at": metadata.get("created_at") or metadata.get("promoted_at"),
                "dataset_id": metrics.get("dataset_id") if metrics else metadata.get("dataset_id"),
                "has_metrics": metrics is not None,
                "metadata": metadata,
            }
        )

    # Sort by creation time (newest first)
    baselines.sort(key=lambda b: str(b.get("created_at") or ""), reverse=True)

    return baselines


def compare_experiments(
    run1_path: Path,
    run2_path: Path,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare two experiment runs.

    Args:
        run1_path: Path to first experiment run directory
        run2_path: Path to second experiment run directory
        dataset_id: Optional dataset ID for validation

    Returns:
        Comparison dictionary with deltas

    Raises:
        FileNotFoundError: If metrics files are missing
        ValueError: If dataset_id mismatch
    """
    from podcast_scraper.evaluation.comparator import compute_delta, load_metrics

    metrics1_path = run1_path / "metrics.json"
    metrics2_path = run2_path / "metrics.json"

    if not metrics1_path.exists():
        raise FileNotFoundError(f"Metrics not found for run 1: {metrics1_path}")
    if not metrics2_path.exists():
        raise FileNotFoundError(f"Metrics not found for run 2: {metrics2_path}")

    metrics1 = load_metrics(metrics1_path)
    metrics2 = load_metrics(metrics2_path)

    # Validate dataset_id if provided
    if dataset_id:
        if metrics1.get("dataset_id") != dataset_id:
            got_id = metrics1.get("dataset_id")
            raise ValueError(f"Run 1 dataset mismatch: expected '{dataset_id}', " f"got '{got_id}'")
        if metrics2.get("dataset_id") != dataset_id:
            got_id = metrics2.get("dataset_id")
            raise ValueError(f"Run 2 dataset mismatch: expected '{dataset_id}', " f"got '{got_id}'")

    deltas: Dict[str, Any] = {}

    # Compare intrinsic metrics
    intrinsic1 = metrics1.get("intrinsic", {})
    intrinsic2 = metrics2.get("intrinsic", {})

    # Cost deltas
    cost1 = intrinsic1.get("cost", {})
    cost2 = intrinsic2.get("cost", {})
    if cost1.get("total_cost_usd") is not None and cost2.get("total_cost_usd") is not None:
        deltas["cost_total_usd"] = compute_delta(
            cost1.get("total_cost_usd"), cost2.get("total_cost_usd")
        )

    # Performance deltas
    perf1 = intrinsic1.get("performance", {})
    perf2 = intrinsic2.get("performance", {})
    if perf1.get("avg_latency_ms") is not None and perf2.get("avg_latency_ms") is not None:
        deltas["avg_latency_ms"] = compute_delta(
            perf1.get("avg_latency_ms"), perf2.get("avg_latency_ms")
        )

    # Gate regressions
    gates1 = intrinsic1.get("gates", {})
    gates2 = intrinsic2.get("gates", {})
    gate_regressions = []
    for gate_name in [
        "boilerplate_leak_rate",
        "speaker_label_leak_rate",
        "truncation_rate",
    ]:
        rate1 = gates1.get(gate_name, 0.0)
        rate2 = gates2.get(gate_name, 0.0)
        if rate1 > rate2:
            gate_regressions.append(gate_name)
    deltas["gate_regressions"] = gate_regressions  # type: ignore[assignment]

    # vs_reference deltas (if both have same references)
    vs_ref1 = metrics1.get("vs_reference")
    vs_ref2 = metrics2.get("vs_reference")
    if vs_ref1 and vs_ref2 and isinstance(vs_ref1, dict) and isinstance(vs_ref2, dict):
        for ref_id in vs_ref1.keys():
            if ref_id in vs_ref2:
                rougeL1 = vs_ref1[ref_id].get("rougeL_f1")
                rougeL2 = vs_ref2[ref_id].get("rougeL_f1")
                if rougeL1 is not None and rougeL2 is not None:
                    deltas[f"rougeL_f1_vs_{ref_id}"] = compute_delta(rougeL1, rougeL2)

    return {
        "run1_id": metrics1.get("run_id"),
        "run2_id": metrics2.get("run_id"),
        "dataset_id": metrics1.get("dataset_id"),
        "deltas": deltas,
    }


def generate_history_report(
    runs: List[Dict[str, Any]],
    dataset_id: Optional[str] = None,
    metric_name: str = "rougeL_f1",
) -> str:
    """Generate a historical trend report for a specific metric.

    Args:
        runs: List of run metadata dictionaries
        dataset_id: Optional dataset ID to filter runs
        metric_name: Metric name to track (e.g., "rougeL_f1", "avg_latency_ms")

    Returns:
        Formatted Markdown report
    """
    lines = []
    lines.append("# Historical Trend Report")
    lines.append("")
    if dataset_id:
        lines.append(f"**Dataset:** `{dataset_id}`")
    lines.append(f"**Metric:** `{metric_name}`")
    lines.append("")

    # Filter runs by dataset if specified
    if dataset_id:
        runs = [r for r in runs if r.get("dataset_id") == dataset_id]

    if not runs:
        lines.append("No runs found.")
        return "\n".join(lines)

    lines.append("## Trend Data")
    lines.append("")
    lines.append("| Run ID | Date | Metric Value |")
    lines.append("|--------|------|--------------|")

    for run in runs[:20]:  # Limit to 20 most recent
        run_id = run.get("run_id", "unknown")
        created_at = run.get("created_at", "")
        path = run.get("path")

        # Try to extract metric value
        metric_value = "N/A"
        if path:
            metrics_path = path / "metrics.json"
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    # Try to extract metric (supports nested paths)
                    extracted_value = _extract_metric_from_dict(metrics, metric_name)
                    if extracted_value is not None:
                        if isinstance(extracted_value, (int, float)):
                            metric_value = f"{extracted_value:.4f}"
                        else:
                            metric_value = str(extracted_value)
                except Exception:
                    pass

        # Format date
        date_str = created_at[:10] if created_at and len(created_at) >= 10 else "unknown"

        lines.append(f"| `{run_id}` | {date_str} | {metric_value} |")

    lines.append("")

    return "\n".join(lines)


def _extract_metric_from_dict(metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
    """Extract metric value from metrics dictionary (supports nested paths)."""
    parts = metric_name.split(".")
    value = metrics

    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)  # type: ignore[assignment]
        else:
            return None

        if value is None:
            return None

    if isinstance(value, (int, float)):
        return float(value)

    return None
