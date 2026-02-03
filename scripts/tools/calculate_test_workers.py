#!/usr/bin/env python3
"""Calculate optimal number of pytest workers based on available memory and CPU.

This script helps prevent memory exhaustion by calculating the maximum number
of parallel workers that can safely run based on:
- Available system memory
- CPU core count
- Estimated memory per worker (varies by test type)
- Platform-specific considerations (Mac vs Linux)

Usage:
    python scripts/tools/calculate_test_workers.py [--test-type TYPE] \
        [--min-workers N] [--max-workers N]

Examples:
    # Default calculation (conservative, assumes integration/E2E tests)
    python scripts/tools/calculate_test_workers.py

    # For unit tests (lower memory per worker)
    python scripts/tools/calculate_test_workers.py --test-type unit

    # For integration tests (default)
    python scripts/tools/calculate_test_workers.py --test-type integration

    # For E2E tests (higher memory per worker)
    python scripts/tools/calculate_test_workers.py --test-type e2e
"""

import argparse
import os
import sys
from typing import Optional

try:
    import psutil
except ImportError:
    print("2", file=sys.stderr)  # Fallback to 2 workers
    sys.exit(0)  # Exit gracefully, don't fail if psutil not available


# Memory estimates per worker (in GB)
# These are conservative estimates based on observed usage
# E2E tests are more memory-intensive due to full pipeline execution
MEMORY_PER_WORKER = {
    "unit": 0.1,  # ~100 MB per unit test worker
    "integration": 1.5,  # ~1.5 GB per integration test worker (ML models)
    "e2e": 2.5,  # ~2.5 GB per E2E test worker (full pipeline, more conservative)
    "default": 1.5,  # Default to integration test estimate
}

# Reserve memory for system (GB)
# Increased reserve for E2E tests to prevent system freezes
SYSTEM_RESERVE_GB = 4.0  # Reserve 4 GB for system operations
SYSTEM_RESERVE_E2E_GB = 8.0  # Reserve 8 GB for E2E tests (more conservative)

# CPU-based limits
MIN_WORKERS = 1
MAX_WORKERS_CPU = 8  # Cap at 8 workers even if memory allows more
MAX_WORKERS_E2E = 4  # Cap E2E tests at 4 workers to prevent system freezes
CPU_RESERVE = 2  # Reserve 2 cores for system


def get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    try:
        mem = psutil.virtual_memory()
        return mem.available / (1024**3)
    except Exception:
        # Fallback: assume 8 GB if we can't detect
        return 8.0


def get_cpu_count() -> int:
    """Get CPU core count."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def is_macos() -> bool:
    """Check if running on macOS."""
    return sys.platform == "darwin"


def calculate_workers(
    test_type: str = "default",
    min_workers: int = MIN_WORKERS,
    max_workers: Optional[int] = None,
) -> int:
    """Calculate optimal number of workers based on memory and CPU.

    Args:
        test_type: Type of tests ("unit", "integration", "e2e", or "default")
        min_workers: Minimum number of workers (default: 1)
        max_workers: Maximum number of workers (default: None, uses CPU-based limit)

    Returns:
        Optimal number of workers
    """
    # Get memory per worker for this test type
    memory_per_worker_gb = MEMORY_PER_WORKER.get(test_type, MEMORY_PER_WORKER["default"])

    # Get system resources
    available_memory_gb = get_available_memory_gb()
    cpu_count = get_cpu_count()

    # Calculate memory-based limit
    # Use higher reserve for E2E tests to prevent system freezes
    reserve_gb = SYSTEM_RESERVE_E2E_GB if test_type == "e2e" else SYSTEM_RESERVE_GB
    usable_memory_gb = max(0, available_memory_gb - reserve_gb)
    memory_based_workers = int(usable_memory_gb / memory_per_worker_gb)

    # Calculate CPU-based limit
    # Reserve 2 cores for system
    # Use lower cap for E2E tests to prevent system freezes
    max_workers_cpu = MAX_WORKERS_E2E if test_type == "e2e" else MAX_WORKERS_CPU
    cpu_based_workers = min(max(MIN_WORKERS, cpu_count - CPU_RESERVE), max_workers_cpu)

    # Use the more restrictive limit (memory or CPU)
    workers = min(memory_based_workers, cpu_based_workers)

    # Apply min/max constraints
    if max_workers is None:
        max_workers = max_workers_cpu
    workers = max(min_workers, min(workers, max_workers))

    # On macOS, be more conservative (reduce by 1-2 workers)
    # Macs often have less available RAM due to system overhead
    # For E2E tests, reduce by 2 on macOS to be extra safe
    if is_macos():
        reduction = 2 if test_type == "e2e" else 1
        workers = max(min_workers, workers - reduction)

    return workers


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate optimal number of pytest workers based on system resources"
    )
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "e2e", "default"],
        default="default",
        help="Type of tests to run (affects memory estimates)",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=MIN_WORKERS,
        help=f"Minimum number of workers (default: {MIN_WORKERS})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help=f"Maximum number of workers (default: {MAX_WORKERS_CPU})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed calculation information",
    )

    args = parser.parse_args()

    # Calculate workers
    workers = calculate_workers(
        test_type=args.test_type,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
    )

    # Print verbose info if requested
    if args.verbose:
        available_memory_gb = get_available_memory_gb()
        cpu_count = get_cpu_count()
        memory_per_worker_gb = MEMORY_PER_WORKER.get(args.test_type, MEMORY_PER_WORKER["default"])
        # Use same reserve calculation as in calculate_workers
        reserve_gb = SYSTEM_RESERVE_E2E_GB if args.test_type == "e2e" else SYSTEM_RESERVE_GB
        usable_memory_gb = max(0, available_memory_gb - reserve_gb)
        memory_based = int(usable_memory_gb / memory_per_worker_gb)
        max_workers_cpu = MAX_WORKERS_E2E if args.test_type == "e2e" else MAX_WORKERS_CPU
        cpu_based = min(max(MIN_WORKERS, cpu_count - CPU_RESERVE), max_workers_cpu)

        print("System Resources:", file=sys.stderr)
        print(f"  CPU cores: {cpu_count}", file=sys.stderr)
        print(f"  Available memory: {available_memory_gb:.2f} GB", file=sys.stderr)
        reserve_msg = (
            f"  Usable memory (after {reserve_gb} GB reserve): " f"{usable_memory_gb:.2f} GB"
        )
        print(reserve_msg, file=sys.stderr)
        platform_name = "macOS" if is_macos() else "Linux/Other"
        print(f"  Platform: {platform_name}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Test Configuration:", file=sys.stderr)
        print(f"  Test type: {args.test_type}", file=sys.stderr)
        print(f"  Memory per worker: {memory_per_worker_gb:.2f} GB", file=sys.stderr)
        print("", file=sys.stderr)
        print("Worker Calculation:", file=sys.stderr)
        print(f"  Memory-based limit: {memory_based} workers", file=sys.stderr)
        print(f"  CPU-based limit: {cpu_based} workers", file=sys.stderr)
        print(f"  Selected: {workers} workers", file=sys.stderr)
        if is_macos():
            reduction = 2 if args.test_type == "e2e" else 1
            print(f"  (Reduced by {reduction} for macOS)", file=sys.stderr)

    # Output just the number (for Makefile to use)
    print(workers)


if __name__ == "__main__":
    main()
