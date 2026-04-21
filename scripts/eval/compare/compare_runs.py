#!/usr/bin/env python3
"""Compare two experiment runs.

This script compares any two experiment runs and generates a comparison report.

Usage:
    python scripts/eval/compare_runs.py --run1 <run_id1> --run2 <run_id2>
        [--dataset-id <dataset_id>]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from podcast_scraper.evaluation.history import compare_experiments, find_all_runs
from podcast_scraper.evaluation.reporter import (
    generate_comparison_report,
    print_report,
    save_report,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare two experiment runs")
    parser.add_argument(
        "--run1",
        type=str,
        required=True,
        help="First run ID (from data/eval/runs/)",
    )
    parser.add_argument(
        "--run2",
        type=str,
        required=True,
        help="Second run ID (from data/eval/runs/)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Optional dataset ID for validation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for comparison report (default: prints to console)",
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

    # Find runs
    runs = find_all_runs()
    run1_dict = next((r for r in runs if r["run_id"] == args.run1), None)
    run2_dict = next((r for r in runs if r["run_id"] == args.run2), None)

    if not run1_dict:
        logger.error(f"Run 1 not found: {args.run1}")
        logger.info("Available runs:")
        for r in runs[:10]:
            logger.info(f"  - {r['run_id']}")
        sys.exit(1)

    if not run2_dict:
        logger.error(f"Run 2 not found: {args.run2}")
        logger.info("Available runs:")
        for r in runs[:10]:
            logger.info(f"  - {r['run_id']}")
        sys.exit(1)

    run1_path = run1_dict["path"]
    run2_path = run2_dict["path"]

    # Compare
    try:
        comparison = compare_experiments(
            run1_path=run1_path,
            run2_path=run2_path,
            dataset_id=args.dataset_id,
        )

        # Generate report
        report = generate_comparison_report(comparison)

        if args.output:
            save_report(report, Path(args.output))
        else:
            print_report(report)

        logger.info("Comparison complete")

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
