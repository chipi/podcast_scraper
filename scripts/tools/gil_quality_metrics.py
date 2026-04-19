#!/usr/bin/env python3
"""Compute PRD-017 GIL quality metrics over ``gi.json`` artifacts (optional enforcement).

Usage (from project root)::

    python scripts/tools/gil_quality_metrics.py path/to/run/metadata
    python scripts/tools/gil_quality_metrics.py path/to/run --enforce --strict-schema

Exits 0 when reporting only, or when ``--enforce`` and all PRD thresholds pass.
Exits 1 on enforce failure or load errors when ``--fail-on-errors``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from podcast_scraper.gi.quality_metrics import (
        compute_gil_quality_metrics,
        enforce_prd017_thresholds,
    )
except ImportError:
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root / "src"))
    from podcast_scraper.gi.quality_metrics import (
        compute_gil_quality_metrics,
        enforce_prd017_thresholds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PRD-017 GIL quality metrics over .gi.json files (file scan, no DB)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories containing .gi.json",
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
        help="Exit 1 if PRD-017 thresholds are not met (use min-* to tune)",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit 1 if any artifact fails to load or validate",
    )
    parser.add_argument(
        "--min-extraction-coverage",
        type=float,
        default=0.80,
        help="Min share of artifacts with ≥1 insight and ≥1 quote (default: 0.80)",
    )
    parser.add_argument(
        "--min-grounded-insight-rate",
        type=float,
        default=0.90,
        help="Min share of insights with grounded=true (default: 0.90)",
    )
    parser.add_argument(
        "--min-quote-validity-rate",
        type=float,
        default=0.95,
        help="Min share of quotes with valid span + timestamps (default: 0.95)",
    )
    parser.add_argument(
        "--min-avg-insights",
        type=float,
        default=5.0,
        help="Min mean insights per artifact (default: 5.0)",
    )
    parser.add_argument(
        "--min-avg-quotes",
        type=float,
        default=10.0,
        help="Min mean quotes per artifact (default: 10.0)",
    )
    args = parser.parse_args()

    m = compute_gil_quality_metrics(args.paths, strict_schema=args.strict_schema)
    payload = m.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("GIL quality metrics")
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
        ok, failures = enforce_prd017_thresholds(
            m,
            min_extraction_coverage=args.min_extraction_coverage,
            min_grounded_insight_rate=args.min_grounded_insight_rate,
            min_quote_validity_rate=args.min_quote_validity_rate,
            min_avg_insights=args.min_avg_insights,
            min_avg_quotes=args.min_avg_quotes,
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
