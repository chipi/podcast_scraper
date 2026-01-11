#!/usr/bin/env python3
"""
Profile individual E2E tests to identify memory-intensive tests.

This script runs each E2E test individually and measures memory usage,
helping identify which tests consume the most memory.

Usage:
    python scripts/tools/profile_e2e_test_memory.py [--test-file FILE] [--top-n N]

Examples:
    # Profile all E2E tests
    python scripts/tools/profile_e2e_test_memory.py

    # Profile specific test file
    python scripts/tools/profile_e2e_test_memory.py --test-file test_ml_models_e2e.py

    # Show top 10 memory consumers
    python scripts/tools/profile_e2e_test_memory.py --top-n 10
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

    # Use pytest's collect-only output to get test nodeids directly
    # Use sys.executable to use the same Python interpreter as the script
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/e2e/",
        "--collect-only",
        "-m",
        "e2e and not analysis",
    ]
    if test_file:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            f"tests/e2e/{test_file}",
            "--collect-only",
            "-m",
            "e2e and not analysis",
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error collecting tests: {result.stderr}")
        sys.exit(1)

    # Parse test nodeids from collect-only output
    # Pytest outputs test nodeids directly in format: file::Class::test_name or file::test_name
    tests = []
    for line in result.stdout.split("\n"):
        line = line.strip()
        # Match test nodeid format: tests/e2e/file.py::Class::test_name
        # or tests/e2e/file.py::test_name
        if "::" in line and line.startswith("tests/e2e/") and ".py" in line:
            # Filter out summary lines like "12 tests collected"
            if "test" in line.lower() and "collected" not in line.lower():
                tests.append(line)

    return sorted(set(tests))  # Remove duplicates and sort


def profile_test(test_path: str) -> Dict[str, any]:
    """Profile a single test's memory usage."""
    print(f"  Profiling: {test_path}...", end=" ", flush=True)

    # Get system memory before
    mem_before = psutil.virtual_memory().used / (1024**3)

    # Start process
    start_time = time.time()
    cmd = [
        sys.executable,
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

    # Get psutil Process object for memory monitoring
    try:
        psutil_process = psutil.Process(process.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # Process already finished or access denied
        process.wait()
        return {
            "test_path": test_path,
            "peak_memory_mb": 0,
            "peak_memory_gb": 0,
            "duration": 0,
            "exit_code": process.returncode,
            "passed": process.returncode == 0,
            "memory_delta_gb": 0,
            "max_children": 0,
        }

    # Monitor memory
    peak_memory_mb = 0
    max_children = 0

    while process.poll() is None:
        try:
            # Get process memory
            mem_info = psutil_process.memory_info()
            mem_mb = mem_info.rss / (1024**2)

            # Get children memory
            children_mem_mb = 0
            children_count = 0
            try:
                for child in psutil_process.children(recursive=True):
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
