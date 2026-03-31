#!/usr/bin/env python3
"""Compute PRD-019-oriented KG metrics over ``kg.json`` artifacts (optional enforcement).

Usage (from project root)::

    python scripts/tools/kg_quality_metrics.py path/to/run/metadata
    python scripts/tools/kg_quality_metrics.py path/to/run --enforce --strict-schema
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from podcast_scraper.kg.quality_metrics import (
        compute_kg_quality_metrics,
        enforce_prd019_thresholds,
    )
except ImportError:
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root / "src"))
    from podcast_scraper.kg.quality_metrics import (
        compute_kg_quality_metrics,
        enforce_prd019_thresholds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PRD-019 KG quality metrics over .kg.json files (file scan, no DB)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories containing .kg.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON only",
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="Validate each artifact with strict JSON Schema before scoring",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit 1 if PRD-019 thresholds are not met (use --min-* to tune)",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit 1 if any path fails to load or validate",
    )
    parser.add_argument(
        "--min-artifacts",
        type=int,
        default=1,
        help="Minimum successfully scored artifacts (default: 1)",
    )
    parser.add_argument(
        "--min-avg-nodes",
        type=float,
        default=1.0,
        help="Minimum mean nodes per artifact (default: 1.0)",
    )
    parser.add_argument(
        "--min-avg-edges",
        type=float,
        default=0.0,
        help="Minimum mean edges per artifact (default: 0.0)",
    )
    parser.add_argument(
        "--min-extraction-coverage",
        type=float,
        default=1.0,
        help="Min share of artifacts with non-empty extraction block (default: 1.0)",
    )
    args = parser.parse_args()

    m = compute_kg_quality_metrics(args.paths, strict_schema=args.strict_schema)
    payload = m.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("KG quality metrics (PRD-019-oriented)")
        print("=" * 50)
        for k, v in payload.items():
            if k == "errors":
                continue
            print(f"  {k}: {v}")
        if m.errors:
            print("\nErrors:")
            for e in m.errors:
                print(f"  - {e}")

    if args.fail_on_errors and m.errors:
        return 1

    if args.enforce:
        ok, failures = enforce_prd019_thresholds(
            m,
            min_artifacts=args.min_artifacts,
            min_avg_nodes=args.min_avg_nodes,
            min_avg_edges=args.min_avg_edges,
            min_extraction_coverage=args.min_extraction_coverage,
        )
        if not ok:
            if not args.json:
                print("\nEnforce failures:")
                for line in failures:
                    print(f"  - {line}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
