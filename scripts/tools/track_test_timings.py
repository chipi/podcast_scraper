#!/usr/bin/env python3
"""Track test execution times for unit, integration, and e2e tests.

This script runs each test suite separately, captures execution times,
and stores them in a JSON file for historical comparison.

Usage:
    python scripts/tools/track_test_timings.py
    # Or via Makefile:
    make test-track
    make test-track-view
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Path to store timing data
TIMINGS_FILE = Path(__file__).parent.parent.parent / "reports" / "test-timings.json"


def run_test_suite(suite_name: str, make_target: str) -> tuple[bool, float]:
    """Run a test suite and return (success, elapsed_time).

    Args:
        suite_name: Human-readable name (e.g., "unit", "integration", "e2e")
        make_target: Makefile target to run (e.g., "test-unit", "test-integration")

    Returns:
        Tuple of (success: bool, elapsed_time: float in seconds)
    """
    print(f"\n{'=' * 60}")
    print(f"Running {suite_name} tests ({make_target})...")
    print(f"{'=' * 60}")

    start_time = time.time()
    try:
        result = subprocess.run(
            ["make", make_target],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=False,  # Show output in real-time
            check=True,
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0
        print(f"\n✓ {suite_name} tests completed in {elapsed:.2f}s")
        return success, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {suite_name} tests failed after {elapsed:.2f}s (exit code: {e.returncode})")
        return False, elapsed
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n⚠ {suite_name} tests interrupted after {elapsed:.2f}s")
        return False, elapsed


def load_timings() -> list[dict]:
    """Load existing timing data from JSON file.

    Returns:
        List of timing records (empty list if file doesn't exist)
    """
    if not TIMINGS_FILE.exists():
        return []

    try:
        with open(TIMINGS_FILE, "r") as f:
            data = json.load(f)
            return data.get("runs", [])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not parse timings file: {e}")
        return []


def save_timings(runs: list[dict]) -> None:
    """Save timing data to JSON file.

    Args:
        runs: List of timing records to save
    """
    TIMINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "description": "Test execution timing history",
            "format_version": "1.0",
            "last_updated": datetime.now().isoformat(),
        },
        "runs": runs,
    }

    with open(TIMINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Timings saved to: {TIMINGS_FILE}")


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


def format_timestamp(iso_str: str) -> str:
    """Format ISO timestamp to readable format.

    Args:
        iso_str: ISO format timestamp string

    Returns:
        Formatted timestamp string
    """
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return iso_str


def print_summary(runs: list[dict], current_run: dict) -> None:
    """Print summary comparing current run to last successful run.

    Args:
        runs: All historical runs (current_run is already appended)
        current_run: Current run data
    """
    baseline = get_last_successful_run(runs, exclude_current=True)

    if not baseline:
        print("\n" + "=" * 60)
        print("Summary (no previous successful run to compare):")
        print("=" * 60)
        print(f"  Unit:       {current_run['unit_seconds']:.2f}s")
        print(f"  Integration: {current_run['integration_seconds']:.2f}s")
        print(f"  E2E:        {current_run['e2e_seconds']:.2f}s")
        print(f"  Total:      {current_run['total_seconds']:.2f}s")
        return

    print("\n" + "=" * 60)
    print("Summary (vs last successful run):")
    print("=" * 60)
    baseline_time = format_timestamp(baseline.get("timestamp", ""))
    print(f"Baseline: {baseline_time}")
    print()

    for suite in ["unit", "integration", "e2e", "total"]:
        key = f"{suite}_seconds"
        current = current_run[key]
        baseline_val = baseline.get(key, 0)

        if baseline_val > 0:
            diff = current - baseline_val
            diff_pct = (diff / baseline_val * 100) if baseline_val > 0 else 0
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(
                f"  {suite.capitalize():12} {current:7.2f}s  {symbol} {diff:+.2f}s ({diff_pct:+.1f}%)"
            )
        else:
            print(f"  {suite.capitalize():12} {current:7.2f}s")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 if all tests passed, 1 otherwise)
    """
    parser = argparse.ArgumentParser(
        description="Track test execution times for unit, integration, and e2e tests"
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="View timing history instead of running tests",
    )
    parser.add_argument(
        "--last",
        type=int,
        help="Show only last N runs (for --view mode)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare current run to previous run (for --view mode)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics for all runs (for --view mode)",
    )

    args = parser.parse_args()

    # If --view flag, delegate to view script
    if args.view:
        import subprocess
        import sys

        view_script = Path(__file__).parent / "view_test_timings.py"
        cmd = [sys.executable, str(view_script)]
        if args.last:
            cmd.extend(["--last", str(args.last)])
        if args.compare:
            cmd.append("--compare")
        if args.stats:
            cmd.append("--stats")
        return subprocess.run(cmd).returncode

    print("Test Timing Tracker")
    print("=" * 60)
    print(f"Timings will be saved to: {TIMINGS_FILE}")

    # Load existing timings
    runs = load_timings()

    # Run each test suite
    results = {}
    all_passed = True

    for suite_name, make_target in [
        ("unit", "test-unit"),
        ("integration", "test-integration"),
        ("e2e", "test-e2e"),
    ]:
        success, elapsed = run_test_suite(suite_name, make_target)
        results[suite_name] = {"success": success, "elapsed": elapsed}
        if not success:
            all_passed = False

    # Calculate totals
    total_elapsed = sum(r["elapsed"] for r in results.values())

    # Create current run record
    current_run = {
        "timestamp": datetime.now().isoformat(),
        "unit_seconds": results["unit"]["elapsed"],
        "unit_success": results["unit"]["success"],
        "integration_seconds": results["integration"]["elapsed"],
        "integration_success": results["integration"]["success"],
        "e2e_seconds": results["e2e"]["elapsed"],
        "e2e_success": results["e2e"]["success"],
        "total_seconds": total_elapsed,
        "all_passed": all_passed,
    }

    # Append to runs
    runs.append(current_run)

    # Keep only last 50 runs (prevent file from growing too large)
    if len(runs) > 50:
        runs = runs[-50:]
        print("\n⚠ Kept only last 50 runs (removed older entries)")

    # Save timings
    save_timings(runs)

    # Print summary
    print_summary(runs, current_run)

    # Return exit code
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
