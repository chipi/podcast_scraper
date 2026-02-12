#!/usr/bin/env python3
"""
Analyze test suite memory usage and resource consumption.

This script helps identify memory leaks, excessive resource usage, and
optimization opportunities in the test suite.

Usage:
    python scripts/tools/analyze_test_memory.py [--test-target TARGET] [--max-workers N]

Examples:
    # Analyze default test target
    python scripts/tools/analyze_test_memory.py

    # Analyze with limited workers
    python scripts/tools/analyze_test_memory.py --max-workers 4

    # Analyze specific test target
    python scripts/tools/analyze_test_memory.py --test-target test-unit
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    print("❌ Error: psutil not installed. Install with: pip install psutil")
    sys.exit(1)


def get_system_info() -> Dict[str, Any]:
    """Get system resource information."""
    return {
        "cpu_count": os.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "memory_used_gb": psutil.virtual_memory().used / (1024**3),
        "memory_percent": psutil.virtual_memory().percent,
    }


def monitor_process_memory(pid: int, interval: float = 1.0) -> List[Dict[str, float]]:
    """Monitor memory usage of a process and its children."""
    samples = []
    try:
        process = psutil.Process(pid)
        while process.is_running():
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

            samples.append(
                {
                    "timestamp": time.time(),
                    "process_memory_mb": mem_mb,
                    "children_memory_mb": children_mem_mb,
                    "children_count": children_count,
                    "total_memory_mb": total_mem_mb,
                }
            )

            time.sleep(interval)
    except psutil.NoSuchProcess:
        pass
    except KeyboardInterrupt:
        pass

    return samples


def run_test_with_monitoring(test_target: str, max_workers: Optional[int] = None) -> Dict[str, Any]:
    """Run tests and monitor memory usage."""
    print(f"🔍 Analyzing memory usage for: {test_target}")
    print(f"   Max workers: {max_workers or 'auto'}")
    print()

    # Get system info before
    system_before = get_system_info()
    print("📊 System Resources (Before):")
    cpu_info = (
        f"{system_before['cpu_count']} (logical: {system_before['cpu_count_logical']}, "
        f"physical: {system_before['cpu_count_physical']})"
    )
    print(f"   CPU cores: {cpu_info}")
    mem_info = (
        f"{system_before['memory_used_gb']:.2f} GB / {system_before['memory_total_gb']:.2f} GB "
        f"({system_before['memory_percent']:.1f}% used)"
    )
    print(f"   Memory: {mem_info}")
    print(f"   Available: {system_before['memory_available_gb']:.2f} GB")
    print()

    # Build make command
    cmd = ["make", test_target]
    if max_workers:
        # Override parallelism by setting environment variable
        env = os.environ.copy()
        env["PYTEST_XDIST_WORKERS"] = str(max_workers)
        # Note: This requires Makefile to respect the env var, or we need to modify the command
        print(f"⚠️  Note: max_workers={max_workers} may not be respected by Makefile")
        print(f"   Consider modifying Makefile to use: -n {max_workers}")
        print()

    print("🚀 Starting test execution...")
    print(f"   Command: {' '.join(cmd)}")
    print()

    # Start process
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Monitor memory
    memory_samples = monitor_process_memory(process.pid, interval=2.0)

    # Wait for completion
    stdout, _ = process.communicate()
    end_time = time.time()
    duration = end_time - start_time

    # Get system info after
    system_after = get_system_info()

    # Analyze memory samples
    if memory_samples:
        max_memory_mb = max(s["total_memory_mb"] for s in memory_samples)
        avg_memory_mb = sum(s["total_memory_mb"] for s in memory_samples) / len(memory_samples)
        max_children = max(s["children_count"] for s in memory_samples)
    else:
        max_memory_mb = 0
        avg_memory_mb = 0
        max_children = 0

    return {
        "test_target": test_target,
        "duration_seconds": duration,
        "exit_code": process.returncode,
        "system_before": system_before,
        "system_after": system_after,
        "memory_samples": memory_samples,
        "max_memory_mb": max_memory_mb,
        "avg_memory_mb": avg_memory_mb,
        "max_children": max_children,
        "stdout": stdout,
    }


def print_analysis(results: Dict[str, Any]) -> None:
    """Print memory analysis results."""
    print()
    print("=" * 80)
    print("📈 Memory Analysis Results")
    print("=" * 80)
    print()

    print(f"Test Target: {results['test_target']}")
    print(f"Duration: {results['duration_seconds']:.1f} seconds")
    print(f"Exit Code: {results['exit_code']}")
    print()

    print("💾 Memory Usage:")
    peak_gb = results["max_memory_mb"] / 1024
    print(f"   Peak Memory: {results['max_memory_mb']:.2f} MB ({peak_gb:.2f} GB)")
    avg_gb = results["avg_memory_mb"] / 1024
    print(f"   Average Memory: {results['avg_memory_mb']:.2f} MB ({avg_gb:.2f} GB)")
    print(f"   Max Worker Processes: {results['max_children']}")
    print()

    system_before = results["system_before"]
    system_after = results["system_after"]

    print("🔄 System Resource Changes:")
    memory_delta = system_after["memory_used_gb"] - system_before["memory_used_gb"]
    print(f"   Memory Delta: {memory_delta:+.2f} GB")
    mem_after_info = (
        f"{system_after['memory_used_gb']:.2f} GB / {system_after['memory_total_gb']:.2f} GB "
        f"({system_after['memory_percent']:.1f}% used)"
    )
    print(f"   Memory After: {mem_after_info}")
    print()

    # Recommendations
    print("💡 Recommendations:")

    total_memory_gb = system_before["memory_total_gb"]
    peak_memory_gb = results["max_memory_mb"] / 1024

    if peak_memory_gb > total_memory_gb * 0.8:
        critical_msg = (
            f"   ⚠️  CRITICAL: Peak memory ({peak_memory_gb:.2f} GB) exceeds 80% of total "
            f"({total_memory_gb:.2f} GB)"
        )
        print(critical_msg)
        print("      Consider reducing parallelism or optimizing test memory usage")

    if results["max_children"] > system_before["cpu_count"]:
        warning_msg = (
            f"   ⚠️  WARNING: Worker count ({results['max_children']}) exceeds CPU cores "
            f"({system_before['cpu_count']})"
        )
        print(warning_msg)
        suggested_workers = system_before["cpu_count"] - 2
        print(f"      Consider using: -n {suggested_workers} (reserve 2 cores)")

    if peak_memory_gb > 8:
        print(f"   ⚠️  WARNING: Peak memory ({peak_memory_gb:.2f} GB) is very high")
        print("      Consider reducing parallelism or investigating memory leaks")

    # Check for memory growth
    if len(results["memory_samples"]) > 10:
        early_avg = sum(s["total_memory_mb"] for s in results["memory_samples"][:5]) / 5
        late_avg = sum(s["total_memory_mb"] for s in results["memory_samples"][-5:]) / 5
        growth = late_avg - early_avg
        if growth > 500:  # 500 MB growth
            print(f"   ⚠️  WARNING: Memory growth detected ({growth:.0f} MB)")
            print("      This may indicate a memory leak")

    print()

    # Memory samples summary
    if results["memory_samples"]:
        print("📊 Memory Usage Over Time (sample points):")
        sample_count = min(10, len(results["memory_samples"]))
        step = max(1, len(results["memory_samples"]) // sample_count)
        for i in range(0, len(results["memory_samples"]), step):
            sample = results["memory_samples"][i]
            elapsed = sample["timestamp"] - results["memory_samples"][0]["timestamp"]
            print(
                f"   {elapsed:6.1f}s: {sample['total_memory_mb']:7.1f} MB "
                f"(process: {sample['process_memory_mb']:6.1f} MB, "
                f"children: {sample['children_memory_mb']:6.1f} MB, "
                f"workers: {sample['children_count']})"
            )
        print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze test suite memory usage and resource consumption"
    )
    parser.add_argument(
        "--test-target",
        default="test-unit",
        help="Makefile test target to analyze (default: test-unit)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers (overrides Makefile setting)",
    )

    args = parser.parse_args()

    # Check if psutil is available (already imported at top of file)
    if "psutil" not in sys.modules:
        print("❌ Error: psutil not installed. Install with: pip install psutil")
        sys.exit(1)

    # Run analysis
    results = run_test_with_monitoring(args.test_target, args.max_workers)

    # Print results
    print_analysis(results)

    # Exit with test exit code
    sys.exit(results["exit_code"])


if __name__ == "__main__":
    main()
