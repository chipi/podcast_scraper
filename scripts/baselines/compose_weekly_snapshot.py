#!/usr/bin/env python3
"""Compose a weekly baseline snapshot from a setup probe + an experiment
run + a benchmark run, write it as ``data/baselines/baseline-YYYY-WNN.json``.

Idempotent: re-running on the same ISO week overwrites the same file
(no append). Designed to be called from
``.github/workflows/nightly-data-baseline.yml``.

Usage::

    .venv/bin/python scripts/baselines/compose_weekly_snapshot.py \\
        --probe probe.json \\
        --experiment-run-dir data/eval/runs/<run_id> \\
        --benchmark-run-dir data/eval/runs/<run_id> \\
        [--output data/baselines/baseline-2026-W26.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _iso_week_id(when: Optional[datetime] = None) -> str:
    """``YYYY-WNN`` per ISO 8601 week date."""
    dt = when or datetime.now(timezone.utc)
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


def _load_optional_json(p: Optional[Path]) -> Dict[str, Any]:
    if p is None or not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)[:200]}


def _run_summary(run_dir: Optional[Path]) -> Dict[str, Any]:
    """Pull (run_id, metrics, fingerprint_hash) from an eval run dir.

    Defensive — missing fields don't fail the snapshot."""
    if run_dir is None or not run_dir.is_dir():
        return {"present": False}
    metrics = _load_optional_json(run_dir / "metrics.json")
    fp = _load_optional_json(run_dir / "fingerprint.json")
    return {
        "present": True,
        "run_id": run_dir.name,
        "metrics": metrics,
        "fingerprint_hash": fp.get("fingerprint_hash"),
    }


def compose(
    probe_path: Optional[Path],
    experiment_run_dir: Optional[Path],
    benchmark_run_dir: Optional[Path],
) -> Dict[str, Any]:
    return {
        "snapshot_version": 1,
        "week_id": _iso_week_id(),
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "setup_probe": _load_optional_json(probe_path),
        "experiment_run": _run_summary(experiment_run_dir),
        "benchmark_run": _run_summary(benchmark_run_dir),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, default=None)
    p.add_argument("--experiment-run-dir", type=Path, default=None)
    p.add_argument("--benchmark-run-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    snap = compose(args.probe, args.experiment_run_dir, args.benchmark_run_dir)
    output = args.output
    if output is None:
        baselines_dir = Path(__file__).resolve().parents[2] / "data" / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)
        output = baselines_dir / f"baseline-{snap['week_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
