#!/usr/bin/env python3
"""Benchmark orchestrator for running experiments across multiple datasets.

This script:
- Runs experiments on all specified datasets
- Compares results against baselines
- Generates comprehensive comparison reports
- Supports smoke mode (fast subset) and full mode

This implements RFC-041 Phase 2: Benchmark Runner.

Usage:
    python scripts/eval/run_benchmark.py --datasets dataset1,dataset2 --baseline baseline_id
    python scripts/eval/run_benchmark.py --smoke --baseline baseline_id  # Fast smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from podcast_scraper.evaluation.config import load_experiment_config
from podcast_scraper.evaluation.history import find_all_baselines
from podcast_scraper.evaluation.reporter import save_report

logger = logging.getLogger(__name__)


def find_datasets(datasets_dir: Path = Path("data/eval/datasets")) -> List[str]:
    """Find all available datasets.

    Args:
        datasets_dir: Directory containing dataset JSON files

    Returns:
        List of dataset IDs (without .json extension)
    """
    if not datasets_dir.exists():
        return []

    datasets = []
    for dataset_file in datasets_dir.glob("*.json"):
        dataset_id = dataset_file.stem
        datasets.append(dataset_id)

    return sorted(datasets)


def run_benchmark(
    experiment_config_path: Path,
    datasets: List[str],
    baseline_id: str,
    reference_ids: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run benchmark across multiple datasets.

    Args:
        experiment_config_path: Path to experiment config YAML
        datasets: List of dataset IDs to run
        baseline_id: Baseline ID for comparison
        reference_ids: Optional list of reference IDs for evaluation
        output_dir: Optional output directory for benchmark results

    Returns:
        Benchmark results dictionary
    """
    if output_dir is None:
        output_dir = Path("data/eval/benchmarks") / Path(experiment_config_path).stem

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running benchmark: {experiment_config_path}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Baseline: {baseline_id}")
    logger.info(f"Output: {output_dir}")

    # Load experiment config
    cfg = load_experiment_config(str(experiment_config_path))

    # Run experiment for each dataset
    results = {}
    for dataset_id in datasets:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiment on dataset: {dataset_id}")
        logger.info(f"{'='*80}")

        # Create a temporary config with this dataset
        dataset_cfg = cfg.model_copy(deep=True)
        dataset_cfg.data.dataset_id = dataset_id
        dataset_cfg.id = f"{cfg.id}_{dataset_id}"

        # Import and run experiment
        # Import here to avoid circular import issues
        import importlib.util

        run_experiment_path = Path(__file__).parent / "run_experiment.py"
        spec = importlib.util.spec_from_file_location("run_experiment_mod", run_experiment_path)
        run_experiment_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_experiment_mod)

        try:
            run_experiment_mod.run_experiment(
                cfg=dataset_cfg,
                baseline_id=baseline_id,
                reference_ids=reference_ids,
            )

            # Load results
            run_id = dataset_cfg.id
            run_path = Path("data/eval/runs") / run_id

            if run_path.exists():
                metrics_path = run_path / "metrics.json"
                comparison_path = run_path / "comparisons" / f"vs_{baseline_id}.json"

                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    comparison = None
                    if comparison_path.exists():
                        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

                    results[dataset_id] = {
                        "run_id": run_id,
                        "metrics": metrics,
                        "comparison": comparison,
                        "status": "success",
                    }
                else:
                    results[dataset_id] = {
                        "status": "error",
                        "error": "Metrics not found",
                    }
            else:
                results[dataset_id] = {
                    "status": "error",
                    "error": "Run directory not found",
                }

        except Exception as e:
            logger.error(f"Failed to run experiment on {dataset_id}: {e}", exc_info=True)
            results[dataset_id] = {
                "status": "error",
                "error": str(e),
            }

    # Generate summary report
    summary = generate_benchmark_summary(results, baseline_id, experiment_config_path)
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Generate human-readable report
    report = generate_benchmark_report(summary, results)
    report_path = output_dir / "benchmark_report.md"
    save_report(report, report_path)

    logger.info(f"\n{'='*80}")
    logger.info(f"Benchmark complete: {output_dir}")
    logger.info(f"{'='*80}")

    return summary


def generate_benchmark_summary(
    results: Dict[str, Dict[str, Any]],
    baseline_id: str,
    experiment_config_path: Path,
) -> Dict[str, Any]:
    """Generate benchmark summary.

    Args:
        results: Dictionary of dataset_id -> result
        baseline_id: Baseline ID used
        experiment_config_path: Path to experiment config

    Returns:
        Summary dictionary
    """
    successful = [d for d, r in results.items() if r.get("status") == "success"]
    failed = [d for d, r in results.items() if r.get("status") == "error"]

    # Aggregate metrics
    all_regressions = []
    for dataset_id, result in results.items():
        if result.get("status") == "success":
            comparison = result.get("comparison")
            if comparison:
                regressions = comparison.get("deltas", {}).get("gate_regressions", [])
                if regressions:
                    all_regressions.append({"dataset": dataset_id, "regressions": regressions})

    return {
        "baseline_id": baseline_id,
        "experiment_config": str(experiment_config_path),
        "total_datasets": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "failed_datasets": failed,
        "regressions": all_regressions,
        "results": results,
    }


def generate_benchmark_report(
    summary: Dict[str, Any],
    results: Dict[str, Dict[str, Any]],
) -> str:
    """Generate human-readable benchmark report.

    Args:
        summary: Benchmark summary dictionary
        results: Detailed results dictionary

    Returns:
        Formatted Markdown report
    """
    lines = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"**Baseline:** `{summary.get('baseline_id', 'unknown')}`")
    lines.append(f"**Experiment Config:** `{summary.get('experiment_config', 'unknown')}`")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Datasets:** {summary.get('total_datasets', 0)}")
    lines.append(f"- **Successful:** {summary.get('successful', 0)}")
    lines.append(f"- **Failed:** {summary.get('failed', 0)}")
    lines.append("")

    # Regressions
    regressions = summary.get("regressions", [])
    if regressions:
        lines.append("## ⚠️ Regressions Detected")
        lines.append("")
        for reg in regressions:
            dataset = reg.get("dataset", "unknown")
            reg_list = reg.get("regressions", [])
            lines.append(f"### {dataset}")
            for r in reg_list:
                lines.append(f"- {r}")
            lines.append("")
    else:
        lines.append("## ✅ No Regressions")
        lines.append("")
        lines.append("All datasets passed quality gates.")
        lines.append("")

    # Per-dataset results
    lines.append("## Per-Dataset Results")
    lines.append("")

    for dataset_id, result in results.items():
        status = result.get("status", "unknown")
        status_icon = "✅" if status == "success" else "❌"

        lines.append(f"### {status_icon} {dataset_id}")
        lines.append("")

        if status == "success":
            metrics = result.get("metrics", {})
            comparison = result.get("comparison")

            # Intrinsic metrics
            intrinsic = metrics.get("intrinsic", {})
            if intrinsic:
                gates = intrinsic.get("gates", {})
                performance = intrinsic.get("performance", {})
                cost = intrinsic.get("cost", {})

                lines.append("**Intrinsic Metrics:**")
                if gates:
                    lines.append(
                        f"- Boilerplate Leak: {gates.get('boilerplate_leak_rate', 0) * 100:.1f}%"
                    )
                    lines.append(f"- Speaker Leak: {gates.get('speaker_leak_rate', 0) * 100:.1f}%")
                    lines.append(f"- Truncation: {gates.get('truncation_rate', 0) * 100:.1f}%")
                if performance:
                    lines.append(f"- Avg Latency: {performance.get('avg_latency_ms', 0):.0f}ms")
                if cost.get("total_cost_usd"):
                    lines.append(f"- Total Cost: ${cost.get('total_cost_usd', 0):.4f}")
                lines.append("")

            # Comparison deltas
            if comparison:
                deltas = comparison.get("deltas", {})
                if deltas:
                    lines.append("**vs Baseline Deltas:**")
                    for key, value in deltas.items():
                        if isinstance(value, list):
                            lines.append(f"- {key}: {value}")
                        elif value is not None:
                            sign = "+" if value >= 0 else ""
                            lines.append(f"- {key}: {sign}{value:.4f}")
                    lines.append("")
        else:
            error = result.get("error", "Unknown error")
            lines.append(f"**Error:** {error}")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmark across multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark on specific datasets
  python scripts/eval/run_benchmark.py \\
    --config data/eval/configs/my_experiment.yaml \\
    --datasets curated_5feeds_smoke_v1,curated_5feeds_benchmark_v1 \\
    --baseline bart_led_baseline_v1

  # Run smoke benchmark (fast subset)
  python scripts/eval/run_benchmark.py \\
    --config data/eval/configs/my_experiment.yaml \\
    --smoke \\
    --baseline bart_led_baseline_v1

  # Run on all datasets
  python scripts/eval/run_benchmark.py \\
    --config data/eval/configs/my_experiment.yaml \\
    --all \\
    --baseline bart_led_baseline_v1
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of dataset IDs (or use --smoke or --all)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke benchmark (smoke test datasets only)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all available datasets",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline ID for comparison",
    )
    parser.add_argument(
        "--reference",
        type=str,
        action="append",
        dest="reference_ids",
        help="Reference ID for evaluation (can be specified multiple times)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for benchmark results (default: data/eval/benchmarks/{config_name})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Determine datasets to run
    if args.smoke:
        # Smoke test: use smoke datasets
        datasets = [d for d in find_datasets() if "smoke" in d.lower()]
        if not datasets:
            logger.error("No smoke test datasets found. Create a dataset with 'smoke' in the name.")
            sys.exit(1)
    elif args.all:
        # All datasets
        datasets = find_datasets()
        if not datasets:
            logger.error("No datasets found in data/eval/datasets/")
            sys.exit(1)
    elif args.datasets:
        # Specific datasets
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        logger.error("Must specify --datasets, --smoke, or --all")
        sys.exit(1)

    logger.info(f"Running benchmark on {len(datasets)} dataset(s): {datasets}")

    # Validate baseline exists
    baselines = find_all_baselines()
    baseline_dict = next((b for b in baselines if b["baseline_id"] == args.baseline), None)
    if not baseline_dict:
        logger.error(f"Baseline not found: {args.baseline}")
        logger.info("Available baselines:")
        for b in baselines[:10]:
            logger.info(f"  - {b['baseline_id']}")
        sys.exit(1)

    # Run benchmark
    try:
        output_dir = Path(args.output_dir) if args.output_dir else None
        run_benchmark(
            experiment_config_path=Path(args.config),
            datasets=datasets,
            baseline_id=args.baseline,
            reference_ids=args.reference_ids,
            output_dir=output_dir,
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
