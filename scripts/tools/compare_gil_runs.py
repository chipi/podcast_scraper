#!/usr/bin/env python3
"""Compare GIL ``gi.json`` outcomes between two pipeline run directories.

Usage (from project root)::

    make compare-gil-runs REF=path/to/reference/run CAND=path/to/candidate/run

    python scripts/tools/compare_gil_runs.py \\
        .test_outputs/benchmark/gil_openai_evidence/run_YYYYMMDD-HHMMSS_xxxxx \\
        .test_outputs/benchmark/gil_ml_evidence/run_YYYYMMDD-HHMMSS_xxxxx

Each path should be a **run root** containing a ``metadata/`` folder with
``*.gi.json``. See ``config/manual/gil_paired_benchmark_*.yaml`` and
``docs/wip/gil-ml-vs-openai-outcome-benchmark.md``.

Exits 0 always (reporting only). For PRD-017 aggregates per run, use
``scripts/tools/gil_quality_metrics.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from podcast_scraper.gi.compare_runs import (
        collect_gil_stats_from_run_root,
        format_text_report,
        paired_episode_rows,
        summarize_agreement,
    )
except ImportError:
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root / "src"))
    from podcast_scraper.gi.compare_runs import (
        collect_gil_stats_from_run_root,
        format_text_report,
        paired_episode_rows,
        summarize_agreement,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare GIL gi.json stats between two pipeline run roots."
    )
    parser.add_argument(
        "reference_run",
        type=Path,
        help="Run directory (contains metadata/*.gi.json)",
    )
    parser.add_argument(
        "candidate_run",
        type=Path,
        help="Second run directory to compare",
    )
    args = parser.parse_args()

    ref_stats = collect_gil_stats_from_run_root(args.reference_run)
    cand_stats = collect_gil_stats_from_run_root(args.candidate_run)
    rows = paired_episode_rows(ref_stats, cand_stats)
    summary = summarize_agreement(rows)
    text = format_text_report(args.reference_run, args.candidate_run, rows, summary)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
