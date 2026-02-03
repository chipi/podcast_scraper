#!/usr/bin/env python3
"""View and compare test execution timing history.

This script provides multiple ways to analyze test execution timing data:
- View timing history as a table
- Compare runs (current vs previous)
- Detect performance warnings (>10% increase) and regressions (>20% increase)
- Show trends (increasing/decreasing/stable over last 5 runs)
- Display statistics (avg, min, max)

Usage:
    python scripts/tools/view_test_timings.py
    python scripts/tools/view_test_timings.py --last 10
    python scripts/tools/view_test_timings.py --compare
    python scripts/tools/view_test_timings.py --regressions  # Legacy mode
    python scripts/tools/view_test_timings.py --warning-threshold 15.0 --regression-threshold 25.0
    python scripts/tools/view_test_timings.py --trends
    python scripts/tools/view_test_timings.py --stats
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Path to timing data
TIMINGS_FILE = Path(__file__).parent.parent.parent / "reports" / "test-timings.json"


def load_timings() -> list[dict]:
    """Load timing data from JSON file.

    Returns:
        List of timing records (empty list if file doesn't exist)
    """
    if not TIMINGS_FILE.exists():
        print(f"Error: Timing file not found: {TIMINGS_FILE}")
        print("Run 'make test-track' first to generate timing data.")
        return []

    try:
        with open(TIMINGS_FILE, "r") as f:
            data = json.load(f)
            return data.get("runs", [])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: Could not parse timings file: {e}")
        return []


def format_timestamp(iso_str: str) -> str:
    """Format ISO timestamp to readable format.

    Args:
        iso_str: ISO format timestamp string

    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return iso_str


def print_table(runs: list[dict], last_n: int = None) -> None:
    """Print timing data as a table.

    Args:
        runs: List of timing records
        last_n: Show only last N runs (None for all)
    """
    if not runs:
        print("No timing data available.")
        return

    if last_n:
        runs = runs[-last_n:]

    print("\n" + "=" * 100)
    print("Test Execution Timing History")
    print("=" * 100)
    print(
        f"{'Timestamp':<20} {'Unit (s)':>10} {'Integration (s)':>15} "
        f"{'E2E (s)':>10} {'Total (s)':>10} {'Status':>8}"
    )
    print("-" * 100)

    for run in runs:
        timestamp = format_timestamp(run.get("timestamp", ""))
        unit = run.get("unit_seconds", 0)
        integration = run.get("integration_seconds", 0)
        e2e = run.get("e2e_seconds", 0)
        total = run.get("total_seconds", 0)
        status = "✓ PASS" if run.get("all_passed", False) else "✗ FAIL"

        print(
            f"{timestamp:<20} {unit:>10.2f} {integration:>15.2f} "
            f"{e2e:>10.2f} {total:>10.2f} {status:>8}"
        )

    print("=" * 100)


def get_last_successful_run(runs: list[dict], exclude_current: bool = False) -> dict | None:
    """Get the last successful run from history.

    Args:
        runs: List of timing records
        exclude_current: If True, exclude the last run (current run)

    Returns:
        Last successful run dict, or None if no successful runs found
    """
    search_runs = runs[:-1] if exclude_current and len(runs) > 1 else runs
    for run in reversed(search_runs):
        if run.get("all_passed", False):
            return run
    return None


def print_comparison(runs: list[dict]) -> None:
    """Print comparison between current run and last successful run.

    Args:
        runs: List of timing records
    """
    if not runs:
        print("No timing data available.")
        return

    current = runs[-1]
    baseline = get_last_successful_run(runs, exclude_current=True)

    if not baseline:
        print("No previous successful run found for comparison.")
        print_table([current])
        return

    print("\n" + "=" * 80)
    print("Timing Comparison: Current vs Last Successful Run")
    print("=" * 80)

    print(f"\nCurrent Run:  {format_timestamp(current.get('timestamp', ''))}")
    print(f"Baseline:    {format_timestamp(baseline.get('timestamp', ''))} (last successful)")
    print()

    for suite in ["unit", "integration", "e2e", "total"]:
        key = f"{suite}_seconds"
        current_val = current.get(key, 0)
        prev_val = baseline.get(key, 0)
        diff = current_val - prev_val
        diff_pct = (diff / prev_val * 100) if prev_val > 0 else 0

        if diff > 0:
            symbol = "↑"
            color = "slower"
        elif diff < 0:
            symbol = "↓"
            color = "faster"
        else:
            symbol = "="
            color = "same"

        print(
            f"  {suite.capitalize():12} "
            f"{current_val:7.2f}s  {symbol} {abs(diff):6.2f}s "
            f"({diff_pct:+.1f}%)  [{color}]"
        )

    print("\n" + "=" * 80)


def print_statistics(runs: list[dict]) -> None:
    """Print statistics about test execution times.

    Args:
        runs: List of timing records
    """
    if not runs:
        return

    print("\n" + "=" * 80)
    print("Statistics (all runs)")
    print("=" * 80)

    for suite in ["unit", "integration", "e2e", "total"]:
        key = f"{suite}_seconds"
        values = [r.get(key, 0) for r in runs if r.get("all_passed", False)]

        if not values:
            continue

        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)

        print(f"\n{suite.capitalize()}:")
        print(f"  Average: {avg:.2f}s")
        print(f"  Min:     {min_val:.2f}s")
        print(f"  Max:     {max_val:.2f}s")
        print(f"  Runs:    {len(values)}")

    print("=" * 80)


def detect_performance_issues(
    runs: list[dict], warning_threshold_pct: float = 10.0, regression_threshold_pct: float = 20.0
) -> tuple[list[dict], list[dict]]:
    """Detect performance warnings and regressions by comparing to last successful run.

    Always compares current run (or each run) against the last successful run
    before it, not just the immediately previous run. This ensures we detect
    issues even if intermediate runs failed.

    Args:
        runs: List of timing records
        warning_threshold_pct: Percentage increase threshold to flag as warning (default: 10%)
        regression_threshold_pct: Percentage increase threshold to flag as regression (default: 20%)

    Returns:
        Tuple of (warnings, regressions), each list contains records with:
        - run_index: Index in runs list
        - timestamp: Run timestamp
        - baseline_timestamp: Baseline (last successful) run timestamp
        - suite: Test suite name
        - current: Current execution time
        - baseline: Baseline (last successful) execution time
        - increase_pct: Percentage increase
        - increase_seconds: Absolute increase in seconds
        - severity: "warning" or "regression"
    """
    if not runs:
        return [], []

    warnings = []
    regressions = []

    # For each run, find the last successful run before it
    for i, current in enumerate(runs):
        if not current.get("all_passed", False):
            continue  # Skip failed runs

        # Find last successful run before this one
        baseline = None
        for j in range(i - 1, -1, -1):
            if runs[j].get("all_passed", False):
                baseline = runs[j]
                break

        if not baseline:
            continue  # No baseline to compare against

        # Compare each suite
        for suite in ["unit", "integration", "e2e", "total"]:
            key = f"{suite}_seconds"
            current_val = current.get(key, 0)
            baseline_val = baseline.get(key, 0)

            if baseline_val == 0:
                continue

            increase_pct = ((current_val - baseline_val) / baseline_val) * 100

            # Create record
            record = {
                "run_index": i,
                "timestamp": current.get("timestamp", ""),
                "baseline_timestamp": baseline.get("timestamp", ""),
                "suite": suite,
                "current": current_val,
                "baseline": baseline_val,
                "increase_pct": increase_pct,
                "increase_seconds": current_val - baseline_val,
            }

            # Categorize: regression > warning threshold
            if increase_pct > regression_threshold_pct:
                record["severity"] = "regression"
                regressions.append(record)
            elif increase_pct > warning_threshold_pct:
                record["severity"] = "warning"
                warnings.append(record)

    return warnings, regressions


def detect_regressions(runs: list[dict], threshold_pct: float = 20.0) -> list[dict]:
    """Legacy function for backward compatibility.

    Args:
        runs: List of timing records
        threshold_pct: Percentage increase threshold (default: 20%)

    Returns:
        List of regression records
    """
    _, regressions = detect_performance_issues(
        runs, warning_threshold_pct=threshold_pct, regression_threshold_pct=threshold_pct
    )
    return regressions


def detect_trends(runs: list[dict], window: int = 5) -> dict[str, str]:
    """Detect trends in test execution times over recent runs.

    Args:
        runs: List of timing records
        window: Number of recent runs to analyze (default: 5)

    Returns:
        Dictionary mapping suite name to trend ("increasing", "decreasing", "stable")
    """
    passed_runs = [r for r in runs if r.get("all_passed", False)]
    if len(passed_runs) < window:
        return {}

    recent_runs = passed_runs[-window:]
    trends = {}

    for suite in ["unit", "integration", "e2e", "total"]:
        key = f"{suite}_seconds"
        values = [r.get(key, 0) for r in recent_runs]

        if len(values) < 2:
            continue

        # Calculate trend: compare first half to second half
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / len(values[:mid])
        second_half_avg = sum(values[mid:]) / len(values[mid:])

        if second_half_avg > first_half_avg * 1.1:  # >10% increase
            trends[suite] = "increasing"
        elif second_half_avg < first_half_avg * 0.9:  # >10% decrease
            trends[suite] = "decreasing"
        else:
            trends[suite] = "stable"

    return trends


def print_performance_issues(
    runs: list[dict],
    warning_threshold_pct: float = 10.0,
    regression_threshold_pct: float = 20.0,
) -> None:
    """Print detected performance warnings and regressions.

    Args:
        runs: List of timing records
        warning_threshold_pct: Percentage increase threshold for warnings (default: 10%)
        regression_threshold_pct: Percentage increase threshold for regressions (default: 20%)
    """
    warnings, regressions = detect_performance_issues(
        runs, warning_threshold_pct, regression_threshold_pct
    )

    if not warnings and not regressions:
        print("\n✓ No performance issues detected")
        return

    # Print warnings first
    if warnings:
        print("\n" + "=" * 80)
        print(f"⚠️  PERFORMANCE WARNINGS (> {warning_threshold_pct}% vs last successful run)")
        print("=" * 80)

        # Group by run
        by_run = {}
        for warning in warnings:
            run_idx = warning["run_index"]
            if run_idx not in by_run:
                by_run[run_idx] = []
            by_run[run_idx].append(warning)

        for run_idx in sorted(by_run.keys()):
            run_warnings = by_run[run_idx]
            timestamp = format_timestamp(run_warnings[0]["timestamp"])
            baseline_timestamp = format_timestamp(run_warnings[0].get("baseline_timestamp", ""))
            print(f"\nRun: {timestamp}")
            print(f"Baseline: {baseline_timestamp} (last successful)")
            print("-" * 80)

            for warning in run_warnings:
                print(
                    f"  {warning['suite'].capitalize():12} "
                    f"{warning['baseline']:7.2f}s → {warning['current']:7.2f}s "
                    f"(+{warning['increase_seconds']:.2f}s, +{warning['increase_pct']:.1f}%)"
                )

        print("=" * 80)

    # Print regressions
    if regressions:
        print("\n" + "=" * 80)
        print(f"🚨 PERFORMANCE REGRESSIONS (> {regression_threshold_pct}% vs last successful run)")
        print("=" * 80)

        # Group by run
        by_run = {}
        for reg in regressions:
            run_idx = reg["run_index"]
            if run_idx not in by_run:
                by_run[run_idx] = []
            by_run[run_idx].append(reg)

        for run_idx in sorted(by_run.keys()):
            run_regs = by_run[run_idx]
            timestamp = format_timestamp(run_regs[0]["timestamp"])
            baseline_timestamp = format_timestamp(run_regs[0].get("baseline_timestamp", ""))
            print(f"\nRun: {timestamp}")
            print(f"Baseline: {baseline_timestamp} (last successful)")
            print("-" * 80)

            for reg in run_regs:
                print(
                    f"  {reg['suite'].capitalize():12} "
                    f"{reg['baseline']:7.2f}s → {reg['current']:7.2f}s "
                    f"(+{reg['increase_seconds']:.2f}s, +{reg['increase_pct']:.1f}%)"
                )

        print("=" * 80)


def print_regressions(runs: list[dict], threshold_pct: float = 20.0) -> None:
    """Legacy function for backward compatibility.

    Args:
        runs: List of timing records
        threshold_pct: Percentage increase threshold (default: 20%)
    """
    print_performance_issues(
        runs, warning_threshold_pct=threshold_pct, regression_threshold_pct=threshold_pct
    )


def print_trends(runs: list[dict]) -> None:
    """Print trends in test execution times.

    Args:
        runs: List of timing records
    """
    trends = detect_trends(runs)

    if not trends:
        return

    print("\n" + "=" * 80)
    print("Trends (last 5 runs)")
    print("=" * 80)

    for suite, trend in trends.items():
        symbol = "↑" if trend == "increasing" else "↓" if trend == "decreasing" else "→"
        print(f"  {suite.capitalize():12} {symbol} {trend}")

    print("=" * 80)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 on success, 1 on error)
    """
    parser = argparse.ArgumentParser(description="View and compare test execution timing history")
    parser.add_argument(
        "--last",
        type=int,
        help="Show only last N runs",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare current run to previous run",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics for all runs",
    )
    parser.add_argument(
        "--regressions",
        action="store_true",
        help="Detect and show performance regressions (legacy: uses single threshold)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Percentage increase threshold for regression detection (legacy, default: 20%%)",
    )
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=10.0,
        help="Percentage increase threshold for warnings (default: 10%%)",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=20.0,
        help="Percentage increase threshold for regressions (default: 20%%)",
    )
    parser.add_argument(
        "--trends",
        action="store_true",
        help="Show trends in test execution times",
    )

    args = parser.parse_args()

    runs = load_timings()
    if not runs:
        return 1

    if args.compare:
        print_comparison(runs)
    elif args.stats:
        print_statistics(runs)
    elif args.regressions:
        # Legacy mode: use single threshold
        print_regressions(runs, threshold_pct=args.threshold)
    elif args.trends:
        print_trends(runs)
    else:
        print_table(runs, last_n=args.last)
        # Always show warnings/regressions and trends in default view if there are enough runs
        if len(runs) >= 2:
            print_performance_issues(
                runs,
                warning_threshold_pct=args.warning_threshold,
                regression_threshold_pct=args.regression_threshold,
            )
            print_trends(runs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
