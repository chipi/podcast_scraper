#!/usr/bin/env python3
"""Corpus diarization / speaker-attribution quality metrics (optional enforcement) — #876.

Run after a local full-corpus re-diarization to validate speaker attribution automatically
(replaces the manual RUNBOOK-876 Step 6 spot-check). Usage (from project root)::

    python scripts/tools/diarization_quality_metrics.py /path/to/corpus
    python scripts/tools/diarization_quality_metrics.py /path/to/corpus --enforce --json

Exits 0 when reporting only, or when ``--enforce`` and all thresholds pass; 1 on enforce
failure. ``--require-num-speakers`` additionally enforces the (currently-known-gap) metadata
propagation of ``diarization_num_speakers``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from podcast_scraper.evaluation.diarization_quality import (
        compute_diarization_quality_metrics,
        enforce_diarization_thresholds,
    )
except ImportError:
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root / "src"))
    from podcast_scraper.evaluation.diarization_quality import (
        compute_diarization_quality_metrics,
        enforce_diarization_thresholds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Corpus root directory to validate")
    parser.add_argument("--json", action="store_true", help="Print full metrics as JSON")
    parser.add_argument("--enforce", action="store_true", help="Exit 1 if thresholds fail")
    parser.add_argument(
        "--require-num-speakers",
        action="store_true",
        help="Also enforce diarization_num_speakers in metadata (known propagation gap)",
    )
    args = parser.parse_args()

    metrics = compute_diarization_quality_metrics(args.corpus)

    if args.json:
        print(json.dumps(metrics, indent=2, default=str))
    else:
        summary = {k: v for k, v in metrics.items() if k != "per_episode"}
        print("Diarization quality:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    passed, failures = enforce_diarization_thresholds(
        metrics, require_num_speakers=args.require_num_speakers
    )
    if failures:
        print("\nThreshold issues:")
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("\n✓ all thresholds pass")

    if args.enforce and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
