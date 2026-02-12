#!/usr/bin/env python3
"""Re-score an existing baseline with vs_reference metrics.

Usage:
    python scripts/eval/rescore_baseline.py \
        --baseline-id baseline_ml_prod_authority_v1 \
        --reference silver_gpt4o_benchmark_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from podcast_scraper.evaluation.scorer import score_run

logger = logging.getLogger(__name__)


def find_reference_path(reference_id: str, dataset_id: str) -> Path:
    """Find reference path (checks baselines and references directories).

    Args:
        reference_id: Reference identifier
        dataset_id: Dataset identifier

    Returns:
        Path to reference directory

    Raises:
        FileNotFoundError: If reference not found
    """
    # Check references directory first (references/{dataset_id}/{reference_id})
    ref_path = Path("data/eval/references") / dataset_id / reference_id
    if ref_path.exists():
        return ref_path

    # Check baselines directory (baselines/{reference_id})
    baseline_path = Path("data/eval/baselines") / reference_id
    if baseline_path.exists():
        return baseline_path

    # Check old benchmarks directory (fallback for older project structure)
    old_baseline_path = Path("benchmarks/baselines") / reference_id
    if old_baseline_path.exists():
        return old_baseline_path

    raise FileNotFoundError(
        f"Reference '{reference_id}' not found. "
        f"Checked: data/eval/references/{dataset_id}/{reference_id}, "
        f"data/eval/baselines/{reference_id}, benchmarks/baselines/{reference_id}"
    )


def _resolve_baseline_dir_and_dataset_id(
    baseline_id: str, baseline_dir: Path | None
) -> tuple[Path, str, dict | None]:
    """Resolve baseline dir and dataset_id from metrics.json or baseline.json."""
    if baseline_dir is None:
        baseline_dir = Path("data/eval/baselines") / baseline_id
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_dir}")
    predictions_path = baseline_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")
    metrics_path = baseline_dir / "metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        dataset_id = existing.get("dataset_id")
    else:
        existing = None
        baseline_json_path = baseline_dir / "baseline.json"
        if baseline_json_path.exists():
            baseline_data = json.loads(baseline_json_path.read_text(encoding="utf-8"))
            dataset_id = baseline_data.get("dataset_id")
        else:
            raise ValueError(
                f"Cannot determine dataset_id. Neither metrics.json nor "
                f"baseline.json found in {baseline_dir}"
            )
    if not dataset_id:
        raise ValueError("dataset_id not found in baseline metadata")
    return baseline_dir, dataset_id, existing


def _collect_reference_paths(reference_ids: list[str], dataset_id: str) -> dict[str, Path]:
    """Resolve and validate reference paths; return ref_id -> Path."""
    reference_paths = {}
    for ref_id in reference_ids:
        try:
            ref_path = find_reference_path(ref_id, dataset_id)
            if not (ref_path / "predictions.jsonl").exists():
                logger.warning("Reference '%s' missing predictions.jsonl, skipping", ref_id)
                continue
            reference_paths[ref_id] = ref_path
            logger.info("Found reference '%s' at %s", ref_id, ref_path)
        except FileNotFoundError as e:
            logger.warning("Reference '%s' not found, skipping: %s", ref_id, e)
    return reference_paths


def _print_vs_ref_summary(metrics: dict, baseline_id: str) -> None:
    """Print vs_reference metrics summary to stdout."""
    vs_ref = metrics.get("vs_reference", {})
    if not vs_ref:
        return
    print("\n" + "=" * 80)
    print("VS REFERENCE METRICS SUMMARY")
    print("=" * 80)
    for ref_id, ref_metrics in vs_ref.items():
        if "error" in ref_metrics:
            print(f"\n{ref_id}: ERROR - {ref_metrics['error']}")
            continue
        print(f"\n{ref_id}:")
        for key, label in [
            ("rougeL_f1", "ROUGE-L F1"),
            ("rouge1_f1", "ROUGE-1 F1"),
            ("rouge2_f1", "ROUGE-2 F1"),
            ("embedding_cosine", "Embedding Similarity"),
            ("coverage_ratio", "Coverage Ratio (ML/Silver)"),
            ("bleu", "BLEU"),
            ("wer", "WER"),
        ]:
            val = ref_metrics.get(key)
            if val is not None:
                fmt = ".3f" if isinstance(val, float) else "s"
                print(f"  {label}: {val:{fmt}}")
    intrinsic = metrics.get("intrinsic", {})
    avg_latency_ms = intrinsic.get("performance", {}).get("avg_latency_ms", 0)
    for ref_id, ref_metrics in vs_ref.items():
        if "error" in ref_metrics:
            continue
        rouge_l = ref_metrics.get("rougeL_f1")
        if rouge_l is not None:
            print(f"\n{'=' * 80}")
            print(f"SUMMARY: {baseline_id} vs {ref_id}")
            print(f"{'=' * 80}")
            print(f"Quality: ~{rouge_l*100:.1f}% of silver (ROUGE-L: {rouge_l:.3f})")
            print(f"Latency: {avg_latency_ms/1000:.1f}s per episode")
            if ref_metrics.get("coverage_ratio") is not None:
                print(
                    f"Coverage: {ref_metrics['coverage_ratio']*100:.1f}% " "token ratio (ML/Silver)"
                )
            print("=" * 80)


def rescore_baseline(
    baseline_id: str,
    reference_ids: list[str],
    baseline_dir: Path | None = None,
) -> None:
    """Re-score an existing baseline with vs_reference metrics.

    Args:
        baseline_id: Baseline identifier
        reference_ids: List of reference IDs for vs_reference metrics
        baseline_dir: Optional baseline directory (default: data/eval/baselines)
    """
    baseline_dir, dataset_id, existing_metrics = _resolve_baseline_dir_and_dataset_id(
        baseline_id, baseline_dir
    )
    logger.info("Re-scoring baseline: %s", baseline_id)
    logger.info("Dataset: %s", dataset_id)
    logger.info("References: %s", ", ".join(reference_ids))
    reference_paths = _collect_reference_paths(reference_ids, dataset_id)
    if not reference_paths:
        raise ValueError("No valid references found")
    metrics_path = baseline_dir / "metrics.json"
    run_id = existing_metrics.get("run_id", baseline_id) if existing_metrics else baseline_id
    metrics = score_run(
        predictions_path=baseline_dir / "predictions.jsonl",
        dataset_id=dataset_id,
        run_id=run_id,
        reference_paths=reference_paths,
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("✓ Metrics updated: %s", metrics_path)
    _print_vs_ref_summary(metrics, baseline_id)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Re-score an existing baseline with vs_reference metrics."
    )
    parser.add_argument(
        "--baseline-id",
        type=str,
        required=True,
        help="Baseline identifier (e.g., 'baseline_ml_prod_authority_v1')",
    )
    parser.add_argument(
        "--reference",
        type=str,
        action="append",
        required=True,
        help="Reference ID for vs_reference metrics (can be specified multiple times)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Optional baseline directory path (default: data/eval/baselines/{baseline_id})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None

    try:
        rescore_baseline(
            baseline_id=args.baseline_id,
            reference_ids=args.reference,
            baseline_dir=baseline_dir,
        )
    except Exception as e:
        logger.error(f"Re-scoring failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
