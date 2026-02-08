#!/usr/bin/env python3
"""Analyze bulk confidence test runs.

This script analyzes collected bulk test run data and generates reports
comparing runs, detecting regressions, and providing actionable insights.

Usage:
    python scripts/tools/analyze_bulk_runs.py \
        --session-id session_20260206_103000 \
        --output-dir .test_outputs/bulk_confidence \
        [--mode basic|comprehensive] \
        [--compare-baseline baseline_id] \
        [--output-format markdown|json|both]

Examples:
    # Basic analysis
    python scripts/tools/analyze_bulk_runs.py \
        --session-id session_20260206_103000 \
        --mode basic

    # Comprehensive analysis with baseline comparison
    python scripts/tools/analyze_bulk_runs.py \
        --session-id session_20260206_103000 \
        --mode comprehensive \
        --compare-baseline planet_money_baseline_v1 \
        --output-format both
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
# from datetime import datetime  # Unused import
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


def load_session_data(session_id: str, output_dir: Path) -> Dict[str, Any]:
    """Load session data from JSON file.

    Args:
        session_id: Session identifier (e.g., '20260208_093757' or 'session_20260208_093757')
        output_dir: Output directory

    Returns:
        Session data dict
    """
    # Normalize session_id (remove 'session_' prefix if present)
    if session_id.startswith("session_"):
        session_id = session_id.replace("session_", "", 1)

    # Try new structure first: sessions/session_{id}/session.json
    session_path = output_dir / "sessions" / f"session_{session_id}" / "session.json"
    if not session_path.exists():
        # Fallback to old structure: session_{id}.json (for backwards compatibility)
        session_path = output_dir / f"session_{session_id}.json"
        if not session_path.exists():
            raise FileNotFoundError(
                f"Session file not found. Tried:\n"
                f"  - {output_dir / 'sessions' / f'session_{session_id}' / 'session.json'}\n"
                f"  - {output_dir / f'session_{session_id}.json'}"
            )

    with open(session_path, "r") as f:
        return json.load(f)


def load_baseline(baseline_id: str, output_dir: Path) -> Dict[str, Any]:
    """Load baseline data.

    Args:
        baseline_id: Baseline identifier
        output_dir: Output directory

    Returns:
        Baseline data dict
    """
    baseline_path = output_dir / "baselines" / baseline_id / "baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    with open(baseline_path, "r") as f:
        return json.load(f)


def analyze_logs_basic(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """Basic log analysis.

    Args:
        run_data: Run data dict

    Returns:
        Basic log analysis
    """
    logs = run_data.get("logs", {})
    return {
        "error_count": len(logs.get("errors", [])),
        "warning_count": len(logs.get("warnings", [])),
        "info_count": logs.get("info_count", 0),
    }


def analyze_logs_comprehensive(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive log analysis.

    Args:
        run_data: Run data dict

    Returns:
        Comprehensive log analysis
    """
    logs = run_data.get("logs", {})
    errors = logs.get("errors", [])
    warnings = logs.get("warnings", [])

    # Categorize errors
    error_categories = defaultdict(int)
    for error in errors:
        error_lower = error.lower()
        if "import" in error_lower or "module" in error_lower:
            error_categories["import_errors"] += 1
        elif "api" in error_lower or "http" in error_lower:
            error_categories["api_errors"] += 1
        elif "file" in error_lower or "path" in error_lower:
            error_categories["file_errors"] += 1
        elif "timeout" in error_lower:
            error_categories["timeout_errors"] += 1
        else:
            error_categories["other_errors"] += 1

    # Categorize warnings
    warning_categories = defaultdict(int)
    for warning in warnings:
        warning_lower = warning.lower()
        if "rate limit" in warning_lower:
            warning_categories["rate_limits"] += 1
        elif "timeout" in warning_lower:
            warning_categories["timeouts"] += 1
        elif "deprecation" in warning_lower:
            warning_categories["deprecations"] += 1
        else:
            warning_categories["other_warnings"] += 1

    return {
        "error_count": len(errors),
        "warning_count": len(warnings),
        "info_count": logs.get("info_count", 0),
        "error_categories": dict(error_categories),
        "warning_categories": dict(warning_categories),
        "sample_errors": errors[:5],  # First 5 errors
        "sample_warnings": warnings[:5],  # First 5 warnings
    }


def analyze_outputs_basic(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """Basic output analysis.

    Args:
        run_data: Run data dict

    Returns:
        Basic output analysis
    """
    outputs = run_data.get("outputs", {})
    return {
        "transcripts": outputs.get("transcripts", 0),
        "metadata": outputs.get("metadata", 0),
        "summaries": outputs.get("summaries", 0),
        "complete": (
            outputs.get("transcripts", 0) > 0
            and outputs.get("metadata", 0) > 0
            and outputs.get("summaries", 0) > 0
        ),
    }


def analyze_outputs_comprehensive(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive output analysis.

    Args:
        run_data: Run data dict

    Returns:
        Comprehensive output analysis
    """
    outputs = run_data.get("outputs", {})
    output_dir = Path(run_data.get("output_dir", ""))

    # Check file validity
    valid_transcripts = 0
    valid_metadata = 0
    valid_summaries = 0

    if output_dir.exists():
        # Check transcript files
        # Note: .cleaned.txt files are for quality tooling later to measure the effect of cleaning.
        # They should NOT be part of any stats or counting - only the primary .txt files count.
        for transcript_file in output_dir.rglob("*.txt"):
            # Skip cleaned files (they're for quality analysis, not counting)
            if transcript_file.name.endswith(".cleaned.txt"):
                continue
            try:
                with open(transcript_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if len(content.strip()) > 0:
                        valid_transcripts += 1
            except Exception:
                pass

        # Check metadata files
        for metadata_file in output_dir.rglob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    json.load(f)  # Validate JSON
                    valid_metadata += 1
            except Exception:
                pass

        for metadata_file in output_dir.rglob("*.yaml"):
            try:
                import yaml

                with open(metadata_file, "r") as f:
                    yaml.safe_load(f)  # Validate YAML
                    valid_metadata += 1
            except Exception:
                pass

        # Check summary files
        # Note: .cleaned.txt files are for quality tooling later to measure the effect of cleaning.
        # They should NOT be part of any stats or counting - only the primary .txt files count.
        for summary_file in output_dir.rglob("*.txt"):
            # Skip cleaned files (they're for quality analysis, not counting)
            if summary_file.name.endswith(".cleaned.txt"):
                continue
            if "summary" in summary_file.name.lower() or "summary" in str(summary_file):
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        if len(content.strip()) > 50:  # Reasonable summary length
                            valid_summaries += 1
                except Exception:
                    pass

    return {
        "transcripts": outputs.get("transcripts", 0),
        "metadata": outputs.get("metadata", 0),
        "summaries": outputs.get("summaries", 0),
        "valid_transcripts": valid_transcripts,
        "valid_metadata": valid_metadata,
        "valid_summaries": valid_summaries,
        "complete": (valid_transcripts > 0 and valid_metadata > 0 and valid_summaries > 0),
        "quality_score": (
            (valid_transcripts + valid_metadata + valid_summaries)
            / max(
                1,
                outputs.get("transcripts", 0)
                + outputs.get("metadata", 0)
                + outputs.get("summaries", 0),
            )
            * 100
        ),
    }


def analyze_performance_basic(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """Basic performance analysis.

    Args:
        run_data: Run data dict

    Returns:
        Basic performance analysis
    """
    return {
        "duration_seconds": run_data.get("duration_seconds", 0),
        "exit_code": run_data.get("exit_code", 1),
        "success": run_data.get("exit_code", 1) == 0,
    }


def analyze_performance_comprehensive(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive performance analysis.

    Args:
        run_data: Run data dict

    Returns:
        Comprehensive performance analysis
    """
    resource_usage = run_data.get("resource_usage", {})
    duration = run_data.get("duration_seconds", 0)
    episodes = run_data.get("episodes_processed", 0)

    return {
        "duration_seconds": duration,
        "exit_code": run_data.get("exit_code", 1),
        "success": run_data.get("exit_code", 1) == 0,
        "episodes_processed": episodes,
        "seconds_per_episode": duration / max(1, episodes),
        "peak_memory_mb": resource_usage.get("peak_memory_mb"),
        "cpu_time_seconds": resource_usage.get("cpu_time_seconds"),
        "cpu_percent": resource_usage.get("cpu_percent"),
    }


def compare_with_baseline(
    runs_data: List[Dict[str, Any]], baseline_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare current runs with baseline.

    Args:
        runs_data: Current runs data
        baseline_data: Baseline data

    Returns:
        Comparison results
    """
    baseline_runs = baseline_data.get("runs", [])
    baseline_map = {r["config_name"]: r for r in baseline_runs}

    comparisons = []
    for run in runs_data:
        config_name = run.get("config_name", "")
        baseline_run = baseline_map.get(config_name)

        if not baseline_run:
            comparisons.append(
                {
                    "config_name": config_name,
                    "status": "no_baseline",
                    "message": "No baseline found for this config",
                }
            )
            continue

        # Compare metrics
        duration_delta = run.get("duration_seconds", 0) - baseline_run.get("duration_seconds", 0)
        duration_percent_change = (
            duration_delta / max(1, baseline_run.get("duration_seconds", 1))
        ) * 100

        exit_code_changed = run.get("exit_code", 1) != baseline_run.get("exit_code", 1)

        episodes_delta = run.get("episodes_processed", 0) - baseline_run.get(
            "episodes_processed", 0
        )

        # Compare resource usage
        resource_delta = {}
        if run.get("resource_usage") and baseline_run.get("resource_usage"):
            run_resources = run["resource_usage"]
            baseline_resources = baseline_run["resource_usage"]
            if run_resources.get("peak_memory_mb") and baseline_resources.get("peak_memory_mb"):
                resource_delta["memory_mb"] = (
                    run_resources["peak_memory_mb"] - baseline_resources["peak_memory_mb"]
                )

        # Determine regression
        is_regression = False
        regression_reasons = []

        if exit_code_changed and run.get("exit_code", 1) != 0:
            is_regression = True
            regression_reasons.append("Exit code changed to non-zero")

        if duration_percent_change > 50:  # 50% slower
            is_regression = True
            regression_reasons.append(f"Duration increased by {duration_percent_change:.1f}%")

        if resource_delta.get("memory_mb", 0) > 500:  # 500MB more memory
            is_regression = True
            regression_reasons.append(f"Memory increased by {resource_delta['memory_mb']:.0f}MB")

        comparisons.append(
            {
                "config_name": config_name,
                "status": "regression" if is_regression else "ok",
                "duration_delta_seconds": round(duration_delta, 2),
                "duration_percent_change": round(duration_percent_change, 2),
                "exit_code_changed": exit_code_changed,
                "episodes_delta": episodes_delta,
                "resource_delta": resource_delta,
                "regression_reasons": regression_reasons,
            }
        )

    return {
        "baseline_id": baseline_data.get("baseline_id"),
        "total_comparisons": len(comparisons),
        "regressions": sum(1 for c in comparisons if c["status"] == "regression"),
        "comparisons": comparisons,
    }


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate statistical metrics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dict with statistical metrics
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std_dev": 0.0,
        }

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def generate_basic_report(
    session_data: Dict[str, Any],
    baseline_comparison: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate basic Markdown report with statistical analysis.

    Args:
        session_data: Session data
        baseline_comparison: Optional baseline comparison

    Returns:
        Markdown report string
    """
    runs = session_data.get("runs", [])
    total_runs = len(runs)
    successful_runs = sum(1 for r in runs if r.get("exit_code", 1) == 0)
    failed_runs = total_runs - successful_runs

    report = []
    report.append("# Bulk E2E Confidence Test Report")
    report.append("")
    report.append(f"**Session ID:** {session_data.get('session_id')}")
    report.append(f"**Start Time:** {session_data.get('start_time', 'N/A')}")
    report.append(f"**End Time:** {session_data.get('end_time', 'N/A')}")
    report.append(f"**Total Configs:** {total_runs}")
    report.append(
        f"**Successful:** {successful_runs} ({successful_runs/max(1,total_runs)*100:.1f}%)"
    )
    report.append(f"**Failed:** {failed_runs} ({failed_runs/max(1,total_runs)*100:.1f}%)")
    report.append(f"**Total Duration:** {session_data.get('total_duration_seconds', 0):.1f}s")
    report.append("")

    # Define filtering functions for errors and warnings (used throughout the report)
    def is_real_error(error_text: str) -> bool:
        """Check if this is a real error or a false positive."""
        error_lower = error_text.lower()

        # PRIORITY 1: Check log level first (most reliable indicator)
        # Check for structured log formats first (most explicit)
        has_error_level = any(
            level in error_lower
            for level in ["level=error", "level=critical", "levelname=error", "levelname=critical"]
        )
        has_warning_level = any(
            level in error_lower
            for level in ["level=warning", "level=warn", "levelname=warning", "levelname=warn"]
        )
        has_info_level = any(level in error_lower for level in ["level=info", "levelname=info"])
        has_debug_level = any(level in error_lower for level in ["level=debug", "levelname=debug"])

        # Check for uppercase log level indicators (standard Python logging format)
        has_error_uppercase = (
            " ERROR " in error_text
            or error_text.startswith("ERROR ")
            or " CRITICAL " in error_text
            or error_text.startswith("CRITICAL ")
        )
        has_warning_uppercase = " WARNING " in error_text or error_text.startswith("WARNING ")
        has_info_uppercase = " INFO " in error_text or error_text.startswith("INFO ")
        has_debug_uppercase = " DEBUG " in error_text or error_text.startswith("DEBUG ")

        # If it's explicitly a WARNING level, it's NOT an error
        if has_warning_level or has_warning_uppercase:
            return False

        # If it's explicitly an ERROR/CRITICAL level, it IS an error
        if has_error_level or has_error_uppercase:
            return True

        # If it's INFO or DEBUG level, it's NOT an error (even if it mentions "error" or "failed")
        if has_info_level or has_info_uppercase or has_debug_level or has_debug_uppercase:
            return False

        # PRIORITY 2: Pattern matching (only if no explicit log level found)
        # Check for Python traceback patterns (always errors)
        if "traceback (most recent call last)" in error_lower or "traceback:" in error_lower:
            return True

        # Exclude false positives
        exclusions = [
            "error_type: none",
            "error_message: none",
            "errors total: 0",
            "no error",
            "error_count: 0",
            "error: none",
            "error: null",
            "'error_type': none",
            "'error_message': none",
            '"error_type": null',
            '"error_message": null',
            "episode statuses:",
            "failed=0",  # Processing job with no failures
            "failed: 0",  # Processing job with no failures
            "failed= 0",  # Processing job with no failures
            "ok=",  # Success indicators (e.g., "ok=3, failed=0")
            "ok:",
            "result: episodes=",  # Result summary lines
            "degradation policy",  # Degradation warnings (e.g., "Saving transcript without summary (degradation policy: ...)")
            "degradation:",  # Degradation logger messages
        ]
        if any(exclusion in error_lower for exclusion in exclusions):
            return False

        # Check for error patterns in other contexts
        error_patterns = [
            "traceback",
            "exception:",
            "error:",
            "failed",
            "failure",
            "critical",
            "fatal",
        ]
        if any(pattern in error_lower for pattern in error_patterns):
            # Exclude success indicators
            if any(
                success_indicator in error_lower
                for success_indicator in [
                    "failed=0",
                    "failed: 0",
                    "failed= 0",
                    "ok=",
                    "ok:",
                    "result: episodes=",
                ]
            ):
                return False
            # Exclude degradation policy warnings (they contain "failed" but are warnings)
            if "degradation" in error_lower and "policy" in error_lower:
                return False  # Degradation warnings are not errors
            return True

        return False

    def is_real_warning(warning_text: str) -> bool:
        """Check if this is a real warning or a false positive."""
        warning_lower = warning_text.lower()

        # PRIORITY 1: Check log level first (most reliable indicator)
        # Check for structured log formats first (most explicit)
        has_warning_level = any(
            level in warning_lower
            for level in ["level=warning", "level=warn", "levelname=warning", "levelname=warn"]
        )
        has_error_level = any(
            level in warning_lower
            for level in ["level=error", "level=critical", "levelname=error", "levelname=critical"]
        )
        has_info_level = any(level in warning_lower for level in ["level=info", "levelname=info"])
        has_debug_level = any(
            level in warning_lower for level in ["level=debug", "levelname=debug"]
        )

        # Check for uppercase log level indicators (standard Python logging format)
        has_warning_uppercase = (
            " WARNING " in warning_text
            or warning_text.startswith("WARNING ")
            or "FutureWarning:" in warning_text
        )
        has_error_uppercase = (
            " ERROR " in warning_text
            or warning_text.startswith("ERROR ")
            or " CRITICAL " in warning_text
            or warning_text.startswith("CRITICAL ")
        )
        has_info_uppercase = " INFO " in warning_text or warning_text.startswith("INFO ")
        has_debug_uppercase = " DEBUG " in warning_text or warning_text.startswith("DEBUG ")

        # If it's explicitly a WARNING level, it IS a warning
        if has_warning_level or has_warning_uppercase:
            return True

        # If it's explicitly an ERROR/CRITICAL level, it's NOT a warning
        if has_error_level or has_error_uppercase:
            return False

        # If it's INFO or DEBUG level, it's NOT a warning (even if it mentions "warning" in parameters)
        if has_info_level or has_info_uppercase or has_debug_level or has_debug_uppercase:
            return False

        # PRIORITY 2: Pattern matching (only if no explicit log level found)
        # Exclude false positives (parameter names, counts, etc.)
        exclusions = [
            "warning_count: 0",
            "warnings total: 0",
            "no warning",
            "suppress_fp16_warning",  # Parameter name, not a warning
            "warning=",  # Parameter assignment
            "warning_count",
            "warnings total",
            "disable_warning",
            "ignore_warning",  # Parameter names
        ]
        if any(exclusion in warning_lower for exclusion in exclusions):
            return False

        # Check for warning patterns in other contexts
        warning_patterns = ["warning:", "warn:", "deprecated", "deprecation"]
        if any(pattern in warning_lower for pattern in warning_patterns):
            # Make sure it's not in a parameter context
            if (
                "suppress" in warning_lower
                or "disable" in warning_lower
                or "ignore" in warning_lower
            ):
                return False
            return True

        return False

    # Analysis and Insights (not duplicating basic stats from per-run table)
    report.append("## Analysis and Insights")
    report.append("")
    report.append(
        "> **Note:** Basic statistics (duration, episodes, memory, errors, warnings) are shown in the [Per-Run Summary](#per-run-summary) table below. This section focuses on additional analysis and insights."
    )
    report.append("")

    # Calculate values for analysis (but don't duplicate them in output)
    durations = [r.get("duration_seconds", 0) for r in runs]
    episodes = [r.get("episodes_processed", 0) for r in runs]
    # total_episodes = sum(episodes)  # Unused variable
    # memory_values = [  # Unused variable
    #     r.get("resource_usage", {}).get("peak_memory_mb", 0)
    #     for r in runs
    #     if r.get("resource_usage", {}).get("peak_memory_mb") is not None
    # ]

    # Count real errors and warnings (filtering false positives) for analysis
    error_counts = []
    warning_counts = []
    for r in runs:
        logs = r.get("logs", {})
        real_errors = [e for e in logs.get("errors", []) if is_real_error(e)]
        real_warnings = [w for w in logs.get("warnings", []) if is_real_warning(w)]
        error_counts.append(len(real_errors))
        warning_counts.append(len(real_warnings))

    total_errors = sum(error_counts)
    total_warnings = sum(warning_counts)

    # Performance variance analysis (insight, not duplication)
    if durations and len(durations) > 1:
        duration_stats = calculate_statistics(durations)
        fastest = min(durations)
        slowest = max(durations)
        if fastest > 0:
            speed_ratio = slowest / fastest
            cv = (
                (duration_stats["std_dev"] / max(0.1, duration_stats["mean"]) * 100)
                if duration_stats["std_dev"] > 0
                else 0
            )
            report.append("### Performance Variance Analysis")
            report.append("")
            if speed_ratio > 2:
                report.append(
                    f"⚠️ **High performance variance detected:** Slowest run is {speed_ratio:.1f}x slower than fastest"
                )
                if cv > 30:
                    report.append(
                        f"  - Coefficient of Variation: {cv:.1f}% (high variance indicates inconsistent performance)"
                    )
            else:
                report.append(
                    f"✅ **Performance is consistent:** Speed variance is {speed_ratio:.1f}x"
                )
                if cv < 10:
                    report.append(
                        f"  - Coefficient of Variation: {cv:.1f}% (low variance indicates stable performance)"
                    )
            report.append("")

    # List all errors and warnings
    if total_errors > 0 or total_warnings > 0:
        report.append("### Error and Warning Details")
        report.append("")

        # Collect all errors and warnings from all runs, filtering false positives
        all_errors = []
        all_warnings = []
        for run in runs:
            logs = run.get("logs", {})
            config_name = run.get("config_name", "unknown")
            for error in logs.get("errors", []):
                if is_real_error(error):
                    all_errors.append((config_name, error))
            for warning in logs.get("warnings", []):
                if is_real_warning(warning):
                    all_warnings.append((config_name, warning))

        if all_errors:
            report.append(f"#### Errors ({len(all_errors)})")
            report.append("")
            for i, (config_name, error) in enumerate(all_errors, 1):
                # Truncate very long errors
                error_display = error[:200] + "..." if len(error) > 200 else error
                report.append(f"{i}. **[{config_name}]** {error_display}")
            report.append("")

        if all_warnings:
            report.append(f"#### Warnings ({len(all_warnings)})")
            report.append("")
            for i, (config_name, warning) in enumerate(all_warnings, 1):
                # Truncate very long warnings
                warning_display = warning[:200] + "..." if len(warning) > 200 else warning
                report.append(f"{i}. **[{config_name}]** {warning_display}")
            report.append("")

    # Output quality analysis (insight, not duplication)
    # Exclude dry-run runs from output quality analysis (they don't produce outputs)
    non_dry_run_runs = [r for r in runs if not r.get("is_dry_run", False)]
    total_transcripts = sum(r.get("outputs", {}).get("transcripts", 0) for r in non_dry_run_runs)
    total_metadata = sum(r.get("outputs", {}).get("metadata", 0) for r in non_dry_run_runs)
    total_summaries = sum(r.get("outputs", {}).get("summaries", 0) for r in non_dry_run_runs)
    # Use non-dry-run episodes for coverage calculation
    non_dry_run_episodes = sum(r.get("episodes_processed", 0) for r in non_dry_run_runs)

    if non_dry_run_episodes > 0:
        transcript_coverage = total_transcripts / non_dry_run_episodes * 100
        metadata_coverage = total_metadata / non_dry_run_episodes * 100
        summary_coverage = total_summaries / non_dry_run_episodes * 100

        # Only report if there are issues
        if transcript_coverage < 100 or metadata_coverage < 100 or summary_coverage < 100:
            report.append("### Output Quality Issues")
            report.append("")
            if transcript_coverage < 100:
                report.append(
                    f"⚠️ **Missing transcripts:** {non_dry_run_episodes - total_transcripts} episodes ({100 - transcript_coverage:.1f}% missing)"
                )
            if metadata_coverage < 100:
                report.append(
                    f"⚠️ **Missing metadata:** {non_dry_run_episodes - total_metadata} episodes ({100 - metadata_coverage:.1f}% missing)"
                )
            if summary_coverage < 100:
                report.append(
                    f"⚠️ **Missing summaries:** {non_dry_run_episodes - total_summaries} episodes ({100 - summary_coverage:.1f}% missing)"
                )
            report.append("")
        else:
            report.append("### Output Quality")
            report.append("")
            report.append(
                "✅ **All outputs complete:** 100% coverage for transcripts, metadata, and summaries"
            )
            report.append("")

    # Check for dry-run mode runs
    dry_run_runs = [r for r in runs if r.get("is_dry_run", False)]
    if dry_run_runs:
        report.append("## Dry-Run Mode Detection")
        report.append("")
        report.append(f"ℹ️ **{len(dry_run_runs)} run(s) detected in dry-run mode.**")
        report.append("")
        report.append(
            "Dry-run mode is a preview mode that plans operations without executing them."
        )
        report.append("Expected behavior for dry-run runs:")
        report.append("- ✅ Exit code: 0 (success)")
        report.append("- 📋 Episodes: 0 (no actual processing)")
        report.append("- 📋 Transcripts: 0 (no files created)")
        report.append("- 📋 Metadata/Summaries: 0 (no files created)")
        report.append("")
        report.append("**Dry-run configs:**")
        for run in dry_run_runs:
            report.append(f"- `{run.get('config_name', 'unknown')}`")
        report.append("")

    # Per-run summary table (primary reference for all basic statistics)
    report.append("## Per-Run Summary")
    report.append("")
    report.append(
        "> **All basic statistics** (duration, episodes, errors, warnings, memory, throughput) are shown in the table below. Use this as the primary reference for per-run metrics."
    )
    report.append("")

    # Build table data first to calculate column widths
    table_rows = []
    headers = [
        "Config",
        "Status",
        "Duration",
        "Episodes",
        "Errors",
        "Warnings",
        "Memory (MB)",
        "Throughput",
    ]

    for run in runs:
        config_name = run.get("config_name", "unknown")
        # Truncate long config names for better table display
        if len(config_name) > 30:
            config_name = config_name[:27] + "..."
        exit_code = run.get("exit_code", 1)
        is_dry_run = run.get("is_dry_run", False)
        # Add dry-run indicator to status
        if is_dry_run:
            status = "✅ (DRY-RUN)" if exit_code == 0 else "❌ (DRY-RUN)"
        else:
            status = "✅" if exit_code == 0 else "❌"
        duration = run.get("duration_seconds", 0)
        episodes = run.get("episodes_processed", 0)
        logs = run.get("logs", {})
        # Filter false positives from error/warning counts in table
        real_errors = [e for e in logs.get("errors", []) if is_real_error(e)]
        real_warnings = [w for w in logs.get("warnings", []) if is_real_warning(w)]
        error_count = len(real_errors)
        warning_count = len(real_warnings)
        memory = run.get("resource_usage", {}).get("peak_memory_mb")
        if memory is not None and memory > 0:
            memory_str = f"{memory:.0f}"
        else:
            memory_str = "N/A"
        throughput = episodes / max(0.1, duration) if duration > 0 else 0
        throughput_str = f"{throughput:.3f}/s" if throughput > 0 else "N/A"
        # For dry-run, show "N/A" for throughput since no actual processing happened
        if is_dry_run:
            throughput_str = "N/A (dry-run)"

        table_rows.append(
            [
                config_name,
                status,
                f"{duration:.1f}s",
                str(episodes),
                str(error_count),
                str(warning_count),
                memory_str,
                throughput_str,
            ]
        )

    # Calculate column widths (ensure minimum width of 3 for separator)
    col_widths = [
        max(len(h), max((len(row[i]) for row in table_rows), default=0), 3)
        for i, h in enumerate(headers)
    ]

    # Build header row with proper spacing
    header_cells = [h.ljust(col_widths[i]) for i, h in enumerate(headers)]
    header_row = "| " + " | ".join(header_cells) + " |"
    report.append(header_row)

    # Build separator row (must match header row pipe positions exactly)
    # Each separator cell is dashes matching the column width, with spaces around pipes
    separator_cells = ["-" * col_widths[i] for i in range(len(headers))]
    separator_row = "| " + " | ".join(separator_cells) + " |"
    report.append(separator_row)

    # Build data rows with proper spacing (must match header/separator alignment)
    for row in table_rows:
        data_cells = [row[i].ljust(col_widths[i]) for i in range(len(headers))]
        data_row = "| " + " | ".join(data_cells) + " |"
        report.append(data_row)

    report.append("")

    # Baseline comparison
    if baseline_comparison:
        report.append("## Baseline Comparison")
        report.append("")
        report.append(f"**Baseline:** {baseline_comparison.get('baseline_id')}")
        regressions = baseline_comparison.get("regressions", 0)
        if regressions > 0:
            report.append(f"⚠️ **{regressions} regression(s) detected**")
        else:
            report.append("✅ **No regressions detected**")
        report.append("")

    return "\n".join(report)


def generate_comprehensive_report(
    session_data: Dict[str, Any],
    baseline_comparison: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate comprehensive Markdown report with deep analysis.

    Args:
        session_data: Session data
        baseline_comparison: Optional baseline comparison

    Returns:
        Markdown report string
    """
    runs = session_data.get("runs", [])
    total_runs = len(runs)
    successful_runs = sum(1 for r in runs if r.get("exit_code", 1) == 0)
    failed_runs = total_runs - successful_runs

    report = []
    report.append("# Bulk E2E Confidence Test Report (Comprehensive Analysis)")
    report.append("")
    report.append(f"**Session ID:** {session_data.get('session_id')}")
    report.append(f"**Start Time:** {session_data.get('start_time', 'N/A')}")
    report.append(f"**End Time:** {session_data.get('end_time', 'N/A')}")
    report.append(f"**Total Configs:** {total_runs}")
    report.append(
        f"**Successful:** {successful_runs} ({successful_runs/max(1,total_runs)*100:.1f}%)"
    )
    report.append(f"**Failed:** {failed_runs} ({failed_runs/max(1,total_runs)*100:.1f}%)")
    report.append(f"**Total Duration:** {session_data.get('total_duration_seconds', 0):.1f}s")
    report.append(
        f"**Average per Config:** {session_data.get('total_duration_seconds', 0) / max(1, total_runs):.1f}s"
    )
    report.append("")

    # Comprehensive Statistical Analysis
    report.append("## Comprehensive Statistical Analysis")
    report.append("")

    # Duration statistics
    durations = [r.get("duration_seconds", 0) for r in runs]
    duration_stats = calculate_statistics(durations)
    report.append("### Duration Analysis")
    report.append("")
    report.append(f"- **Count:** {duration_stats['count']}")
    report.append(f"- **Mean:** {duration_stats['mean']:.1f}s")
    report.append(f"- **Median:** {duration_stats['median']:.1f}s")
    report.append(f"- **Min:** {duration_stats['min']:.1f}s")
    report.append(f"- **Max:** {duration_stats['max']:.1f}s")
    if duration_stats["std_dev"] > 0:
        report.append(f"- **Std Dev:** {duration_stats['std_dev']:.1f}s")
        cv = duration_stats["std_dev"] / max(0.1, duration_stats["mean"]) * 100
        report.append(f"- **Coefficient of Variation:** {cv:.1f}%")
        if cv > 30:
            report.append("  ⚠️ **High variance detected** - performance is inconsistent")
        elif cv < 10:
            report.append("  ✅ **Low variance** - performance is consistent")
    report.append("")

    # Episodes statistics
    episodes = [r.get("episodes_processed", 0) for r in runs]
    episodes_stats = calculate_statistics(episodes)
    total_episodes = sum(episodes)
    report.append("### Episodes Processed Analysis")
    report.append("")
    report.append(f"- **Total Episodes:** {total_episodes}")
    report.append(f"- **Mean per Run:** {episodes_stats['mean']:.1f}")
    report.append(f"- **Median:** {episodes_stats['median']:.1f}")
    report.append(f"- **Min:** {episodes_stats['min']:.0f}")
    report.append(f"- **Max:** {episodes_stats['max']:.0f}")
    if episodes_stats["std_dev"] > 0:
        report.append(f"- **Std Dev:** {episodes_stats['std_dev']:.1f}")
    if total_episodes > 0 and session_data.get("total_duration_seconds", 0) > 0:
        throughput = total_episodes / session_data.get("total_duration_seconds", 1)
        report.append(f"- **Overall Throughput:** {throughput:.3f} episodes/second")
        avg_time_per_episode = session_data.get("total_duration_seconds", 0) / total_episodes
        report.append(f"- **Average Time per Episode:** {avg_time_per_episode:.1f}s")
    report.append("")

    # Resource usage statistics
    memory_values = [
        r.get("resource_usage", {}).get("peak_memory_mb", 0)
        for r in runs
        if r.get("resource_usage", {}).get("peak_memory_mb") is not None
    ]
    cpu_time_values = [
        r.get("resource_usage", {}).get("cpu_time_seconds", 0)
        for r in runs
        if r.get("resource_usage", {}).get("cpu_time_seconds") is not None
    ]
    cpu_percent_values = [
        r.get("resource_usage", {}).get("cpu_percent", 0)
        for r in runs
        if r.get("resource_usage", {}).get("cpu_percent") is not None
    ]

    if memory_values:
        memory_stats = calculate_statistics(memory_values)
        report.append("### Memory Usage Analysis")
        report.append("")
        report.append(f"- **Mean Peak Memory:** {memory_stats['mean']:.0f}MB")
        report.append(f"- **Median:** {memory_stats['median']:.0f}MB")
        report.append(f"- **Min:** {memory_stats['min']:.0f}MB")
        report.append(f"- **Max:** {memory_stats['max']:.0f}MB")
        if memory_stats["std_dev"] > 0:
            report.append(f"- **Std Dev:** {memory_stats['std_dev']:.0f}MB")
        if memory_stats["max"] > 4000:
            report.append("  ⚠️ **High memory usage detected** - may indicate memory leaks")
        report.append("")

    if cpu_time_values:
        cpu_stats = calculate_statistics(cpu_time_values)
        report.append("### CPU Usage Analysis")
        report.append("")
        report.append(f"- **Mean CPU Time:** {cpu_stats['mean']:.1f}s")
        report.append(f"- **Median:** {cpu_stats['median']:.1f}s")
        report.append(f"- **Range:** {cpu_stats['min']:.1f}s - {cpu_stats['max']:.1f}s")
        if cpu_percent_values:
            avg_cpu_percent = statistics.mean(cpu_percent_values)
            report.append(f"- **Average CPU Percent:** {avg_cpu_percent:.1f}%")
        report.append("")

    # Error and warning analysis
    error_counts = [len(r.get("logs", {}).get("errors", [])) for r in runs]
    warning_counts = [len(r.get("logs", {}).get("warnings", [])) for r in runs]
    total_errors = sum(error_counts)
    total_warnings = sum(warning_counts)

    report.append("### Error & Warning Analysis")
    report.append("")
    report.append(f"- **Total Errors:** {total_errors}")
    report.append(f"- **Total Warnings:** {total_warnings}")
    if error_counts:
        error_stats = calculate_statistics(error_counts)
        report.append(f"- **Mean Errors per Run:** {error_stats['mean']:.1f}")
        report.append(f"- **Max Errors in Single Run:** {error_stats['max']:.0f}")
    if warning_counts:
        warning_stats = calculate_statistics(warning_counts)
        report.append(f"- **Mean Warnings per Run:** {warning_stats['mean']:.1f}")
        report.append(f"- **Max Warnings in Single Run:** {warning_stats['max']:.0f}")

    runs_with_errors = sum(1 for c in error_counts if c > 0)
    runs_with_warnings = sum(1 for c in warning_counts if c > 0)
    if total_runs > 0:
        report.append(
            f"- **Runs with Errors:** {runs_with_errors} ({runs_with_errors/total_runs*100:.1f}%)"
        )
        report.append(
            f"- **Runs with Warnings:** {runs_with_warnings} ({runs_with_warnings/total_runs*100:.1f}%)"
        )

    # Error categorization
    all_errors = []
    for r in runs:
        all_errors.extend(r.get("logs", {}).get("errors", []))

    if all_errors:
        error_categories = defaultdict(int)
        for error in all_errors:
            error_lower = error.lower()
            if "import" in error_lower or "module" in error_lower:
                error_categories["Import/Module Errors"] += 1
            elif "api" in error_lower or "http" in error_lower or "request" in error_lower:
                error_categories["API/HTTP Errors"] += 1
            elif "file" in error_lower or "path" in error_lower or "permission" in error_lower:
                error_categories["File/Path Errors"] += 1
            elif "timeout" in error_lower:
                error_categories["Timeout Errors"] += 1
            elif "memory" in error_lower or "out of memory" in error_lower:
                error_categories["Memory Errors"] += 1
            elif "key" in error_lower or "credential" in error_lower:
                error_categories["Authentication Errors"] += 1
            else:
                error_categories["Other Errors"] += 1

        if error_categories:
            report.append("")
            report.append("**Error Categories:**")
            for category, count in sorted(
                error_categories.items(), key=lambda x: x[1], reverse=True
            ):
                report.append(f"  - {category}: {count} ({count/len(all_errors)*100:.1f}%)")
    report.append("")

    # Output quality analysis
    total_transcripts = sum(r.get("outputs", {}).get("transcripts", 0) for r in runs)
    total_metadata = sum(r.get("outputs", {}).get("metadata", 0) for r in runs)
    total_summaries = sum(r.get("outputs", {}).get("summaries", 0) for r in runs)

    report.append("### Output Quality Analysis")
    report.append("")
    report.append(f"- **Total Transcripts Generated:** {total_transcripts}")
    report.append(f"- **Total Metadata Files:** {total_metadata}")
    report.append(f"- **Total Summaries:** {total_summaries}")
    if total_episodes > 0:
        transcript_coverage = total_transcripts / total_episodes * 100
        metadata_coverage = total_metadata / total_episodes * 100
        summary_coverage = total_summaries / total_episodes * 100
        report.append(f"- **Transcript Coverage:** {transcript_coverage:.1f}%")
        report.append(f"- **Metadata Coverage:** {metadata_coverage:.1f}%")
        report.append(f"- **Summary Coverage:** {summary_coverage:.1f}%")

        if transcript_coverage < 100:
            report.append(
                f"  ⚠️ **Missing transcripts:** {total_episodes - total_transcripts} episodes"
            )
        if metadata_coverage < 100:
            report.append(f"  ⚠️ **Missing metadata:** {total_episodes - total_metadata} episodes")
        if summary_coverage < 100:
            report.append(
                f"  ⚠️ **Missing summaries:** {total_episodes - total_summaries} episodes"
            )
    report.append("")

    # Performance insights and recommendations
    report.append("## Performance Insights & Recommendations")
    report.append("")

    if durations:
        fastest = min(durations)
        slowest = max(durations)
        if fastest > 0:
            speed_ratio = slowest / fastest
            if speed_ratio > 3:
                report.append(
                    f"⚠️ **High performance variance:** Slowest run is {speed_ratio:.1f}x slower than fastest"
                )
                report.append(
                    "  - **Recommendation:** Investigate why some configs are significantly slower"
                )
            elif speed_ratio > 2:
                report.append(
                    f"⚠️ **Moderate performance variance:** Slowest run is {speed_ratio:.1f}x slower than fastest"
                )
            else:
                report.append(
                    f"✅ **Performance is consistent:** Speed variance is {speed_ratio:.1f}x"
                )
        report.append("")

    # Efficiency metrics
    if total_episodes > 0 and session_data.get("total_duration_seconds", 0) > 0:
        avg_time_per_episode = session_data.get("total_duration_seconds", 0) / total_episodes
        report.append(f"- **Average Processing Time per Episode:** {avg_time_per_episode:.1f}s")

        # Calculate efficiency per run
        efficiency_scores = []
        for r in runs:
            ep_count = r.get("episodes_processed", 0)
            duration = r.get("duration_seconds", 0)
            if ep_count > 0 and duration > 0:
                efficiency = ep_count / duration  # episodes per second
                efficiency_scores.append(efficiency)

        if efficiency_scores:
            eff_stats = calculate_statistics(efficiency_scores)
            report.append(f"- **Mean Throughput:** {eff_stats['mean']:.3f} episodes/second")
            report.append(
                f"- **Throughput Range:** {eff_stats['min']:.3f} - {eff_stats['max']:.3f} episodes/second"
            )
        report.append("")

    # Resource efficiency
    if memory_values and total_episodes > 0:
        avg_memory_per_episode = statistics.mean(memory_values) / max(1, total_episodes / len(runs))
        report.append(f"- **Average Memory per Episode:** {avg_memory_per_episode:.0f}MB")
        if avg_memory_per_episode > 500:
            report.append("  ⚠️ **High memory usage per episode** - consider optimization")
    report.append("")

    # Key findings summary
    report.append("## Key Findings Summary")
    report.append("")

    # Find slowest/fastest runs
    sorted_runs = sorted(runs, key=lambda r: r.get("duration_seconds", 0), reverse=True)
    if sorted_runs:
        slowest = sorted_runs[0]
        fastest = sorted_runs[-1]
        report.append(
            f"**Slowest Run:** {slowest.get('config_name')} ({slowest.get('duration_seconds', 0):.1f}s)"
        )
        if len(sorted_runs) > 1:
            report.append(
                f"**Fastest Run:** {fastest.get('config_name')} ({fastest.get('duration_seconds', 0):.1f}s)"
            )

    # Find runs with most errors
    error_runs = sorted(runs, key=lambda r: len(r.get("logs", {}).get("errors", [])), reverse=True)
    if error_runs and len(error_runs[0].get("logs", {}).get("errors", [])) > 0:
        most_errors = error_runs[0]
        report.append(
            f"**Most Errors:** {most_errors.get('config_name')} "
            f"({len(most_errors.get('logs', {}).get('errors', []))} errors)"
        )

    # Find runs with resource issues
    if memory_values:
        highest_memory_run = max(
            runs,
            key=lambda r: r.get("resource_usage", {}).get("peak_memory_mb", 0) or 0,
        )
        highest_memory = highest_memory_run.get("resource_usage", {}).get("peak_memory_mb", 0)
        if highest_memory > 0:
            report.append(
                f"**Highest Memory:** {highest_memory_run.get('config_name')} "
                f"({highest_memory:.0f}MB)"
            )

    report.append("")

    # Detailed per-run analysis
    report.append("## Detailed Per-Run Analysis")
    report.append("")

    for i, run in enumerate(runs, 1):
        config_name = run.get("config_name", "unknown")
        exit_code = run.get("exit_code", 1)
        status = "✅ PASS" if exit_code == 0 else "❌ FAIL"

        is_dry_run = run.get("is_dry_run", False)
        status_suffix = " (DRY-RUN)" if is_dry_run else ""
        report.append(f"### Run {i}: {config_name} - {status}{status_suffix}")
        report.append("")

        # Basic metrics
        duration = run.get("duration_seconds", 0)
        episodes = run.get("episodes_processed", 0)
        report.append("#### Execution Metrics")
        report.append("")
        if is_dry_run:
            report.append(
                "ℹ️ **Dry-run mode:** This run planned operations without executing them."
            )
            report.append("")
        report.append(f"- **Duration:** {duration:.1f}s")
        report.append(f"- **Episodes Processed:** {episodes}")
        if is_dry_run:
            report.append("  _(Expected: 0 in dry-run mode)_")
        report.append(f"- **Exit Code:** {exit_code}")
        if episodes > 0 and duration > 0:
            throughput = episodes / duration
            time_per_episode = duration / episodes
            report.append(f"- **Throughput:** {throughput:.3f} episodes/second")
            report.append(f"- **Time per Episode:** {time_per_episode:.1f}s")
        report.append("")

        # Resource usage
        resource_usage = run.get("resource_usage", {})
        if resource_usage.get("peak_memory_mb") or resource_usage.get("cpu_time_seconds"):
            report.append("#### Resource Usage")
            report.append("")
            if resource_usage.get("peak_memory_mb"):
                memory = resource_usage["peak_memory_mb"]
                report.append(f"- **Peak Memory:** {memory:.0f}MB")
                if episodes > 0:
                    memory_per_episode = memory / episodes
                    report.append(f"- **Memory per Episode:** {memory_per_episode:.0f}MB")
            if resource_usage.get("cpu_time_seconds"):
                cpu_time = resource_usage["cpu_time_seconds"]
                report.append(f"- **CPU Time:** {cpu_time:.1f}s")
                if duration > 0:
                    cpu_efficiency = cpu_time / duration * 100
                    report.append(f"- **CPU Efficiency:** {cpu_efficiency:.1f}%")
            if resource_usage.get("cpu_percent"):
                report.append(f"- **Average CPU Percent:** {resource_usage['cpu_percent']:.1f}%")
            report.append("")

        # Logs analysis
        logs = run.get("logs", {})
        error_count = len(logs.get("errors", []))
        warning_count = len(logs.get("warnings", []))
        info_count = logs.get("info_count", 0)

        report.append("#### Log Analysis")
        report.append("")
        report.append(f"- **Errors:** {error_count}")
        report.append(f"- **Warnings:** {warning_count}")
        report.append(f"- **Info Messages:** {info_count}")

        if error_count > 0:
            errors = logs.get("errors", [])
            report.append("")
            report.append("**Error Details:**")
            for j, error in enumerate(errors[:5], 1):
                report.append(f"  {j}. {error[:150]}")
            if len(errors) > 5:
                report.append(f"  ... and {len(errors) - 5} more errors")

        if warning_count > 0:
            warnings = logs.get("warnings", [])
            report.append("")
            report.append("**Warning Details:**")
            for j, warning in enumerate(warnings[:5], 1):
                report.append(f"  {j}. {warning[:150]}")
            if len(warnings) > 5:
                report.append(f"  ... and {len(warnings) - 5} more warnings")
        report.append("")

        # Outputs analysis
        outputs = run.get("outputs", {})
        transcripts = outputs.get("transcripts", 0)
        metadata = outputs.get("metadata", 0)
        summaries = outputs.get("summaries", 0)

        report.append("#### Output Analysis")
        report.append("")
        report.append(f"- **Transcripts Generated:** {transcripts}")
        report.append(f"- **Metadata Files:** {metadata}")
        report.append(f"- **Summaries Generated:** {summaries}")

        if episodes > 0:
            transcript_coverage = transcripts / episodes * 100
            metadata_coverage = metadata / episodes * 100
            summary_coverage = summaries / episodes * 100
            report.append("")
            report.append("**Coverage:**")
            report.append(f"- Transcripts: {transcript_coverage:.1f}%")
            report.append(f"- Metadata: {metadata_coverage:.1f}%")
            report.append(f"- Summaries: {summary_coverage:.1f}%")

            if transcript_coverage < 100 or metadata_coverage < 100 or summary_coverage < 100:
                report.append("")
                report.append("⚠️ **Incomplete outputs detected**")
        report.append("")

        # Performance comparison (if multiple runs)
        if len(runs) > 1:
            avg_duration = duration_stats["mean"]
            if duration > avg_duration * 1.2:
                report.append(
                    f"⚠️ **Slower than average:** {((duration / avg_duration - 1) * 100):.1f}% above mean"
                )
            elif duration < avg_duration * 0.8:
                report.append(
                    f"✅ **Faster than average:** {((1 - duration / avg_duration) * 100):.1f}% below mean"
                )
            report.append("")

    # Baseline comparison
    if baseline_comparison:
        report.append("## Baseline Comparison")
        report.append("")
        report.append(f"**Baseline:** {baseline_comparison.get('baseline_id')}")
        regressions = baseline_comparison.get("regressions", 0)
        report.append(f"**Regressions Detected:** {regressions}")
        report.append("")

        for comparison in baseline_comparison.get("comparisons", []):
            config_name = comparison.get("config_name", "unknown")
            status = comparison.get("status", "unknown")

            if status == "regression":
                report.append(f"### ❌ {config_name} - REGRESSION")
                report.append("")
                for reason in comparison.get("regression_reasons", []):
                    report.append(f"- {reason}")
                report.append("")
            elif status == "ok":
                report.append(f"### ✅ {config_name} - OK")
                duration_change = comparison.get("duration_percent_change", 0)
                if abs(duration_change) > 5:
                    report.append(f"- Duration changed by {duration_change:+.1f}%")
                report.append("")

    return "\n".join(report)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze bulk confidence test runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Session ID (e.g., 'session_20260206_103000')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".test_outputs/bulk_confidence",
        help="Output directory (default: .test_outputs/bulk_confidence)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="basic",
        choices=["basic", "comprehensive"],
        help="Analysis mode (default: basic)",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Baseline ID to compare against (optional)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="both",
        choices=["markdown", "json", "both"],
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load session data
    output_dir = Path(args.output_dir)
    session_data = load_session_data(args.session_id, output_dir)

    # Analyze runs
    runs = session_data.get("runs", [])
    analysis_results = []

    for run in runs:
        if args.mode == "basic":
            analysis = {
                "config_name": run.get("config_name"),
                "logs": analyze_logs_basic(run),
                "outputs": analyze_outputs_basic(run),
                "performance": analyze_performance_basic(run),
            }
        else:  # comprehensive
            analysis = {
                "config_name": run.get("config_name"),
                "logs": analyze_logs_comprehensive(run),
                "outputs": analyze_outputs_comprehensive(run),
                "performance": analyze_performance_comprehensive(run),
            }
        analysis_results.append(analysis)

    # Baseline comparison
    baseline_comparison = None
    if args.compare_baseline:
        try:
            baseline_data = load_baseline(args.compare_baseline, output_dir)
            baseline_comparison = compare_with_baseline(runs, baseline_data)
        except FileNotFoundError as e:
            logger.warning(f"Baseline not found: {e}")

    # Generate reports
    if args.mode == "basic":
        markdown_report = generate_basic_report(session_data, baseline_comparison)
    else:
        markdown_report = generate_comprehensive_report(session_data, baseline_comparison)

    # Determine session directory (new structure: sessions/session_{id}/)
    # Normalize session_id (remove 'session_' prefix if present)
    normalized_session_id = args.session_id
    if normalized_session_id.startswith("session_"):
        normalized_session_id = normalized_session_id.replace("session_", "", 1)

    # Try new structure first: sessions/session_{id}/
    session_dir = output_dir / "sessions" / f"session_{normalized_session_id}"
    if not session_dir.exists():
        # Fallback: assume reports go in output_dir (old structure)
        session_dir = output_dir

    # Save reports in session directory
    session_id = args.session_id
    if args.output_format in ["markdown", "both"]:
        report_path = session_dir / f"report_{normalized_session_id}.md"
        with open(report_path, "w") as f:
            f.write(markdown_report)
        logger.info(f"Markdown report saved: {report_path}")

    if args.output_format in ["json", "both"]:
        # Calculate comprehensive statistics for JSON
        durations = [r.get("duration_seconds", 0) for r in runs]
        episodes = [r.get("episodes_processed", 0) for r in runs]
        memory_values = [
            r.get("resource_usage", {}).get("peak_memory_mb", 0)
            for r in runs
            if r.get("resource_usage", {}).get("peak_memory_mb") is not None
        ]
        error_counts = [len(r.get("logs", {}).get("errors", [])) for r in runs]
        warning_counts = [len(r.get("logs", {}).get("warnings", [])) for r in runs]

        json_report = {
            "session_id": session_id,
            "analysis_mode": args.mode,
            "session_summary": {
                "total_runs": len(runs),
                "successful_runs": sum(1 for r in runs if r.get("exit_code", 1) == 0),
                "failed_runs": sum(1 for r in runs if r.get("exit_code", 1) != 0),
                "total_duration_seconds": session_data.get("total_duration_seconds", 0),
                "start_time": session_data.get("start_time"),
                "end_time": session_data.get("end_time"),
            },
            "statistics": {
                "duration": calculate_statistics(durations),
                "episodes": calculate_statistics(episodes),
                "memory": calculate_statistics(memory_values) if memory_values else None,
                "errors": calculate_statistics(error_counts) if error_counts else None,
                "warnings": calculate_statistics(warning_counts) if warning_counts else None,
            },
            "analysis_results": analysis_results,
            "baseline_comparison": baseline_comparison,
            "raw_runs": runs,  # Include all raw run data for deep analysis
        }
        json_path = session_dir / f"report_{normalized_session_id}.json"
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2)
        logger.info(f"JSON report saved: {json_path}")

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
