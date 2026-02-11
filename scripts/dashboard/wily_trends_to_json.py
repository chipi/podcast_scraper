#!/usr/bin/env python3
"""
Generate reports/wily/trends.json from current radon metrics and previous run's metrics.

Used in CI after wily build: compares current complexity/maintainability (from radon
reports) with the previous run (from metrics history) to produce trend strings and
optionally file-level data when wily diff output is available.

See Issue #424: Code quality trends tracking.
"""

import json
import sys
from pathlib import Path


def _read_json(path: Path, default: object = None) -> object:
    """Read JSON file; return default if missing or invalid."""
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _current_complexity(reports_dir: Path) -> float:
    """Get current cyclomatic complexity average from radon report (radon 5.1 or 6)."""
    path = reports_dir / "complexity.json"
    data = _read_json(path)
    if isinstance(data, dict):
        if "total_average" in data:
            return float(data.get("total_average", 0) or 0)
        # Radon 5.1: dict path -> list of {complexity, ...}
        total = 0.0
        count = 0
        for items in data.values():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "complexity" in item:
                        total += item.get("complexity", 0)
                        count += 1
        return total / count if count else 0.0
    if isinstance(data, list) and data:
        total = sum(item.get("complexity", 0) for item in data if isinstance(item, dict))
        count = len([x for x in data if isinstance(x, dict)])
        return total / count if count else 0.0
    return 0.0


def _current_maintainability(reports_dir: Path) -> float:
    """Get current maintainability index average from radon report (radon 5.1 or 6)."""
    path = reports_dir / "maintainability.json"
    data = _read_json(path)
    if isinstance(data, list) and data:
        mi_values = [item.get("mi", 0) for item in data if isinstance(item, dict) and "mi" in item]
        return sum(mi_values) / len(mi_values) if mi_values else 0.0
    if isinstance(data, dict) and data:
        mi_values = [v.get("mi", 0) for v in data.values() if isinstance(v, dict) and "mi" in v]
        return sum(mi_values) / len(mi_values) if mi_values else 0.0
    return 0.0


def generate_trends(
    reports_dir: Path,
    history_path: Path,
    output_path: Path,
) -> None:
    """
    Write trends.json with complexity_trend, maintainability_trend, and file lists.

    Compares current radon metrics (from reports_dir) with the previous run
    (last entry in history_path). File-level degrading/improving are left empty
    unless we add wily diff parsing later.
    """
    current_cc = _current_complexity(reports_dir)
    current_mi = _current_maintainability(reports_dir)

    prev_cc: float | None = None
    prev_mi: float | None = None
    if history_path.exists():
        try:
            with open(history_path) as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                last = json.loads(lines[-1])
                comp = last.get("metrics", {}).get("complexity", {})
                prev_cc = comp.get("cyclomatic_complexity")
                prev_mi = comp.get("maintainability_index")
        except (json.JSONDecodeError, OSError, IndexError):
            pass

    if prev_cc is not None and prev_cc != 0:
        delta_cc = current_cc - prev_cc
        complexity_trend = f"{delta_cc:+.1f}"
    else:
        complexity_trend = "N/A"

    if prev_mi is not None and prev_mi != 0:
        delta_mi = current_mi - prev_mi
        maintainability_trend = f"{delta_mi:+.1f}"
    else:
        maintainability_trend = "N/A"

    out = {
        "complexity_trend": complexity_trend,
        "maintainability_trend": maintainability_trend,
        "files_degrading": [],
        "files_improving": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"Wrote {output_path}: complexity_trend={complexity_trend}, maintainability_trend={maintainability_trend}"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate wily trends JSON for metrics pipeline")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Reports directory (complexity.json, maintainability.json)",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("metrics/history-ci.jsonl"),
        help="Path to history.jsonl (previous run metrics)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/wily/trends.json"),
        help="Output path for trends.json",
    )
    args = parser.parse_args()
    generate_trends(args.reports_dir, args.history, args.output)


if __name__ == "__main__":
    main()
    sys.exit(0)
