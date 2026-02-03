#!/usr/bin/env python3
"""List all experiment runs and baselines.

This script lists all available runs and baselines with their metadata.

Usage:
    python scripts/eval/list_runs.py [--baselines] [--dataset-id <dataset_id>]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from podcast_scraper.evaluation.history import find_all_baselines, find_all_runs

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="List experiment runs and baselines")
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="List baselines instead of runs",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Filter by dataset ID",
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

    if args.baselines:
        items = find_all_baselines()
        item_type = "baseline"
        id_key = "baseline_id"
    else:
        items = find_all_runs()
        item_type = "run"
        id_key = "run_id"

    # Filter by dataset if specified
    if args.dataset_id:
        items = [i for i in items if i.get("dataset_id") == args.dataset_id]

    if not items:
        logger.info(
            f"No {item_type}s found"
            + (f" for dataset '{args.dataset_id}'" if args.dataset_id else "")
        )
        return

    logger.info(f"Found {len(items)} {item_type}(s):")
    logger.info("")

    for item in items[:50]:  # Limit to 50
        item_id = item.get(id_key, "unknown")
        dataset_id = item.get("dataset_id", "unknown")
        created_at = item.get("created_at", "unknown")
        has_metrics = item.get("has_metrics", False)

        status = "✅" if has_metrics else "⚠️ "
        logger.info(f"{status} {item_id}")
        logger.info(f"   Dataset: {dataset_id}")
        logger.info(f"   Created: {created_at}")
        logger.info("")


if __name__ == "__main__":
    main()
