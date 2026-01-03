#!/usr/bin/env python3
"""
Profile individual E2E tests to identify memory-intensive tests.

This script runs each E2E test individually and measures memory usage,
helping identify which tests consume the most memory.

Usage:
    python scripts/profile_e2e_test_memory.py [--test-file FILE] [--top-n N]

Examples:
    # Profile all E2E tests
    python scripts/profile_e2e_test_memory.py

    # Profile specific test file
    python scripts/profile_e2e_test_memory.py --test-file test_ml_models_e2e.py

    # Show top 10 memory consumers
    python scripts/profile_e2e_test_memory.py --top-n 10
"""

import argparse
import subprocess
import sys
import time
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:
    print("❌ Error: psutil not installed. Install with: pip install psutil")
    sys.exit(1)


def get_test_list(test_file: Optional[str] = None) -> List[str]:
    """Get list of E2E tests to profile."""
    import re

    # Use pytest's JSON output for reliable test collection
    cmd = ["python", "-m", "pytest", "tests/e2e/", "--collect-only", "--collect-only", "-q"]
    if test_file:
        cmd = ["python", "-m", "pytest", f"tests/e2e/{test_file}", "--collect-only", "-q"]

    # First get test files
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error collecting tests: {result.stderr}")
        sys.exit(1)

    # Parse file:count format from -q output
    test_files = []
    for line in result.stdout.split("\n"):
        if "tests/e2e/" in line and ".py" in line:
            match = re.search(r"(tests/e2e/[^\s:]+\.py)", line)
            if match:
                if not test_file or test_file in match.group(1):
                    test_files.append(match.group(1))

    # For each file, get individual test nodeids
    tests = []
    for test_file_path in test_files:
        cmd2 = ["python", "-m", "pytest", test_file_path, "--collect-only"]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)

        # Build context as we parse
        current_class = None
        for line in result2.stdout.split("\n"):
            line = line.strip()
            # Track current class
            if "<Class" in line:
                match = re.search(r"<Class\s+([^>]+)>", line)
                if match:
                    current_class = match.group(1)
            # Collect test functions
            elif "<Function" in line:
                match = re.search(r"<Function\s+([^>]+)>", line)
                if match:
                    func_name = match.group(1)
                    if current_class:
                        test_path = f"{test_file_path}::{current_class}::{func_name}"
                    else:
                        test_path = f"{test_file_path}::{func_name}"
                    tests.append(test_path)

    return sorted(set(tests))  # Remove duplicates and sort


def profile_test(test_path: str) -> Dict[str, any]:
    """Profile a single test's memory usage."""
    print(f"  Profiling: {test_path}...", end=" ", flush=True)

    # Get system memory before
    mem_before = psutil.virtual_memory().used / (1024**3)

    # Start process
    start_time = time.time()
    cmd = [
        "python",
        "-m",
        "pytest",
        test_path,
        "-x",  # Stop on first failure
        "--tb=no",  # No traceback
        "-q",  # Quiet
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Monitor memory
    peak_memory_mb = 0
    max_children = 0

    while process.poll() is None:
        try:
            # Get process memory
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / (1024**2)

            # Get children memory
            children_mem_mb = 0
            children_count = 0
            try:
                for child in process.children(recursive=True):
                    try:
                        child_mem = child.memory_info()
                        children_mem_mb += child_mem.rss / (1024**2)
                        children_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            total_mem_mb = mem_mb + children_mem_mb
            peak_memory_mb = max(peak_memory_mb, total_mem_mb)
            max_children = max(max_children, children_count)

            time.sleep(0.5)  # Sample every 500ms
        except (psutil.NoSuchProcess, KeyboardInterrupt):
            break

    # Wait for completion
    stdout, _ = process.communicate()
    end_time = time.time()
    duration = end_time - start_time

    # Get system memory after
    mem_after = psutil.virtual_memory().used / (1024**3)
    memory_delta = mem_after - mem_before

    # Parse result
    exit_code = process.returncode
    passed = exit_code == 0

    print(f"{'✓' if passed else '✗'} ({peak_memory_mb:.0f} MB, {duration:.1f}s)")

    return {
        "test_path": test_path,
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_gb": peak_memory_mb / 1024,
        "duration": duration,
        "exit_code": exit_code,
        "passed": passed,
        "memory_delta_gb": memory_delta,
        "max_children": max_children,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile individual E2E tests to identify memory-intensive tests"
    )
    parser.add_argument(
        "--test-file",
        help="Profile tests from specific file only (e.g., test_ml_models_e2e.py)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Show top N memory consumers (default: 20)",
    )

    args = parser.parse_args()

    # Get test list
    print("📋 Collecting E2E tests...")
    tests = get_test_list(args.test_file)
    print(f"   Found {len(tests)} tests")
    print()

    if not tests:
        print("❌ No tests found!")
        sys.exit(1)

    # Profile each test
    print("🔍 Profiling tests...")
    print()

    results = []
    for i, test_path in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}]", end=" ")
        try:
            result = profile_test(test_path)
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    if not results:
        print("❌ No test results collected!")
        sys.exit(1)

    # Sort by peak memory
    results.sort(key=lambda x: x["peak_memory_mb"], reverse=True)

    # Print summary
    print()
    print("=" * 80)
    print("📊 Top Memory Consumers")
    print("=" * 80)
    print()

    top_n = min(args.top_n, len(results))
    for i, result in enumerate(results[:top_n], 1):
        status = "✓" if result["passed"] else "✗"
        print(
            f"{i:2d}. {status} {result['peak_memory_gb']:6.2f} GB "
            f"({result['peak_memory_mb']:7.0f} MB) - {result['duration']:5.1f}s - "
            f"{result['test_path']}"
        )

    print()
    print("=" * 80)
    print("📈 Summary Statistics")
    print("=" * 80)
    print()

    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]

    if passed:
        avg_memory = sum(r["peak_memory_mb"] for r in passed) / len(passed)
        max_memory = max(r["peak_memory_mb"] for r in passed)
        print(f"Passed tests: {len(passed)}")
        print(f"  Average peak memory: {avg_memory:.0f} MB ({avg_memory/1024:.2f} GB)")
        print(f"  Maximum peak memory: {max_memory:.0f} MB ({max_memory/1024:.2f} GB)")
        print()

    if failed:
        print(f"Failed tests: {len(failed)}")
        for result in failed:
            print(f"  ✗ {result['test_path']}")
        print()

    # Identify high memory consumers
    high_memory_threshold = 2000  # 2 GB
    high_memory_tests = [r for r in results if r["peak_memory_mb"] > high_memory_threshold]

    if high_memory_tests:
        print(f"⚠️  High memory consumers (> {high_memory_threshold/1024:.1f} GB):")
        for result in high_memory_tests:
            print(f"  - {result['test_path']}: {result['peak_memory_gb']:.2f} GB")
        print()


if __name__ == "__main__":
    main()
