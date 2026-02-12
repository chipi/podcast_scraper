#!/usr/bin/env python3
"""
Generate metrics JSON from test artifacts (JUnit XML, coverage XML, pytest JSON).

This script extracts metrics from test artifacts and generates a structured JSON file
for metrics consumption and trend tracking.

See RFC-025: Test Metrics and Health Tracking
"""

import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_test_metrics(pytest_json_path: Path) -> dict:
    """Extract test health metrics from pytest JSON report."""
    if not pytest_json_path.exists():
        return {}

    with open(pytest_json_path) as f:
        pytest_data = json.load(f)

    summary = pytest_data.get("summary", {})
    tests = pytest_data.get("tests", [])

    # Count flaky tests (tests that passed on rerun)
    flaky_tests = [t for t in tests if t.get("outcome") == "passed" and t.get("rerun") is True]
    flaky_count = len(flaky_tests)

    # Extract flaky test details
    flaky_details = []
    for test in flaky_tests:
        flaky_details.append(
            {
                "name": test.get("nodeid", "unknown"),
                "duration": test.get("duration", 0),
            }
        )

    total = summary.get("total", 0)
    passed = summary.get("passed", 0)

    return {
        "total": total,
        "passed": passed,
        "failed": summary.get("failed", 0),
        "skipped": summary.get("skipped", 0),
        "flaky": flaky_count,
        "flaky_tests": flaky_details,
        "pass_rate": passed / total if total > 0 else 0.0,
    }


def extract_runtime_metrics(pytest_json_path: Path) -> dict:
    """Extract runtime metrics from pytest JSON report."""
    if not pytest_json_path.exists():
        return {}

    with open(pytest_json_path) as f:
        pytest_data = json.load(f)

    duration = pytest_data.get("duration", 0)
    summary = pytest_data.get("summary", {})
    total = summary.get("total", 0)

    return {
        "total": duration,
        "tests_per_second": total / duration if duration > 0 else 0.0,
    }


def extract_coverage_metrics(coverage_xml_path: Path, threshold: float = 80.0) -> dict:
    """Extract coverage metrics from coverage XML report.

    Args:
        coverage_xml_path: Path to coverage.xml file
        threshold: Coverage threshold percentage (default: 80.0)
                   This is used for display and alerting purposes.
                   Note: Threshold is only enforced on COMBINED coverage,
                   not on individual test type coverage (unit/integration/e2e).
    """
    if not coverage_xml_path.exists():
        return {"threshold": threshold}

    tree = ET.parse(coverage_xml_path)  # nosec B314
    root = tree.getroot()

    overall = float(root.attrib.get("line-rate", 0)) * 100
    branch_rate = float(root.attrib.get("branch-rate", 0)) * 100

    by_module = {}
    for package in root.findall(".//package"):
        name = package.attrib.get("name", "unknown")
        line_rate = float(package.attrib.get("line-rate", 0)) * 100
        by_module[name] = line_rate

    return {
        "overall": overall,
        "branch": branch_rate,
        "threshold": threshold,
        "meets_threshold": overall >= threshold,
        "by_module": by_module,
    }


def extract_wily_trends(reports_dir: Path) -> dict:
    """Extract trend metrics from wily trends.json (from CI or wily_trends_to_json.py).

    Returns dict with complexity_trend, maintainability_trend, files_degrading,
    files_improving. Missing or invalid file yields N/A and empty lists.
    """
    trends_path = reports_dir / "wily" / "trends.json"
    if not trends_path.exists():
        return {
            "complexity_trend": "N/A",
            "maintainability_trend": "N/A",
            "files_degrading": [],
            "files_improving": [],
        }
    try:
        with open(trends_path) as f:
            data = json.load(f)
        return {
            "complexity_trend": data.get("complexity_trend", "N/A"),
            "maintainability_trend": data.get("maintainability_trend", "N/A"),
            "files_degrading": data.get("files_degrading", []) or [],
            "files_improving": data.get("files_improving", []) or [],
        }
    except (json.JSONDecodeError, IOError):
        return {
            "complexity_trend": "N/A",
            "maintainability_trend": "N/A",
            "files_degrading": [],
            "files_improving": [],
        }


def _parse_complexity_json(path: Path) -> Optional[float]:
    """Parse radon complexity.json; return average complexity or None."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    if isinstance(data, dict) and "total_average" in data:
        return data.get("total_average", 0)
    if isinstance(data, dict):
        total_cc, count = 0, 0
        for items in data.values():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "complexity" in item:
                        total_cc += item.get("complexity", 0)
                        count += 1
        return total_cc / count if count > 0 else None
    if isinstance(data, list) and data:
        total_cc = sum(item.get("complexity", 0) for item in data if isinstance(item, dict))
        count = len([x for x in data if isinstance(x, dict)])
        return total_cc / count if count > 0 else None
    return None


def _parse_maintainability_json(path: Path) -> Optional[float]:
    """Parse radon maintainability.json; return average MI or None."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    if isinstance(data, list) and data:
        mi_values = [item.get("mi", 0) for item in data if isinstance(item, dict) and "mi" in item]
        return sum(mi_values) / len(mi_values) if mi_values else None
    if isinstance(data, dict) and data:
        mi_values = [v.get("mi", 0) for v in data.values() if isinstance(v, dict) and "mi" in v]
        return sum(mi_values) / len(mi_values) if mi_values else None
    return None


def _parse_docstrings_json(path: Path) -> Optional[float]:
    """Parse docstrings coverage JSON; return coverage_percent or None."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    if isinstance(data, dict):
        return data.get("coverage_percent", 0)
    return None


def _parse_vulture_json(path: Path) -> int:
    """Parse vulture dead-code JSON; return count."""
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except (json.JSONDecodeError, IOError):
        return 0


def _parse_codespell_errors(path: Path) -> int:
    """Parse codespell output; return error line count."""
    if not path.exists():
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        lines = [
            line
            for line in text.split("\n")
            if line.strip() and ":" in line and not line.startswith("#")
        ]
        return len(lines)
    except (IOError, UnicodeDecodeError):
        return 0


def extract_complexity_metrics(reports_dir: Path) -> dict:
    """Extract complexity metrics from radon reports and wily trends."""
    metrics = {}
    complexity_path = reports_dir / "complexity.json"
    if complexity_path.exists():
        val = _parse_complexity_json(complexity_path)
        if val is not None:
            metrics["cyclomatic_complexity"] = val
    mi_path = reports_dir / "maintainability.json"
    if mi_path.exists():
        val = _parse_maintainability_json(mi_path)
        if val is not None:
            metrics["maintainability_index"] = val
    doc_path = reports_dir / "docstrings.json"
    if doc_path.exists():
        val = _parse_docstrings_json(doc_path)
        if val is not None:
            metrics["docstring_coverage"] = val
    metrics["dead_code_count"] = _parse_vulture_json(reports_dir / "vulture.json")
    metrics["spelling_errors_count"] = _parse_codespell_errors(reports_dir / "codespell.txt")
    wily_trends = extract_wily_trends(reports_dir)
    metrics["complexity_trend"] = wily_trends.get("complexity_trend", "N/A")
    metrics["maintainability_trend"] = wily_trends.get("maintainability_trend", "N/A")
    metrics["files_degrading"] = wily_trends.get("files_degrading", [])
    metrics["files_improving"] = wily_trends.get("files_improving", [])

    return metrics


def _extract_tests_from_json(json_path: Path) -> list:
    """Extract tests with durations from pytest JSON report."""
    tests = []
    try:
        with open(json_path) as f:
            pytest_data = json.load(f)

        test_list = pytest_data.get("tests", [])
        for test in test_list:
            test_name = test.get("nodeid", "unknown")
            duration = test.get("duration", 0)
            if duration > 0:  # Only include tests with duration data
                tests.append(
                    {
                        "name": test_name,
                        "duration": duration,
                    }
                )
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.warning(f"Failed to extract tests from {json_path}: {e}")
    return tests


def _extract_tests_from_junit(junit_xml_path: Path) -> list:
    """Extract tests with durations from JUnit XML report."""
    tests = []
    try:
        tree = ET.parse(junit_xml_path)  # nosec B314
        root = tree.getroot()

        for testcase in root.findall(".//testcase"):
            name = testcase.get("name", "unknown")
            classname = testcase.get("classname", "")
            time = float(testcase.get("time", 0))

            # Construct full test name
            if classname:
                full_name = f"{classname}::{name}"
            else:
                full_name = name

            if time > 0:  # Only include tests with duration data
                tests.append(
                    {
                        "name": full_name,
                        "duration": time,
                    }
                )
    except (ET.ParseError, ValueError, OSError) as e:
        logger.warning(f"Failed to extract tests from {junit_xml_path}: {e}")
    return tests


def extract_slowest_tests(reports_dir: Path, top_n: int = 20) -> list:
    """Extract slowest tests from pytest JSON reports or JUnit XML.

    Looks for pytest JSON files first (preferred), falls back to JUnit XML.
    Aggregates tests from multiple JSON files if present.

    Args:
        reports_dir: Directory containing test reports
        top_n: Number of slowest tests to return (default: 20)

    Returns:
        List of dicts with 'name' and 'duration' keys, sorted by duration descending
    """
    tests = []

    # Try pytest JSON files first (preferred - always generated in CI)
    pytest_json_patterns = [
        "pytest.json",
        "pytest-unit.json",
        "pytest-integration.json",
        "pytest-e2e.json",
        "pytest-nightly.json",
        "pytest-e2e-serial.json",
    ]

    for pattern in pytest_json_patterns:
        json_path = reports_dir / pattern
        if json_path.exists():
            extracted = _extract_tests_from_json(json_path)
            tests.extend(extracted)
            logger.debug(f"Extracted {len(extracted)} tests from {pattern}")

    # Fallback to JUnit XML if no JSON files found
    if not tests:
        junit_xml_path = reports_dir / "junit.xml"
        if junit_xml_path.exists():
            tests = _extract_tests_from_junit(junit_xml_path)
            logger.debug(f"Extracted {len(tests)} tests from JUnit XML")

    # Sort by duration (descending) and return top N
    tests.sort(key=lambda x: x["duration"], reverse=True)
    return tests[:top_n]


def load_history(history_path: Path) -> List[Dict[str, Any]]:
    """Load historical metrics from history.jsonl."""
    if not history_path.exists():
        return []

    history = []
    try:
        with open(history_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    history.append(json.loads(line))
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Warning: Could not load history: {e}", file=sys.stderr)
        return []

    return history


def calculate_trends(current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Calculate trends compared to previous run."""
    if not previous:
        return {}

    trends = {}

    # Runtime change
    current_runtime = current.get("metrics", {}).get("runtime", {}).get("total", 0)
    prev_runtime = previous.get("metrics", {}).get("runtime", {}).get("total", 0)
    if prev_runtime > 0:
        change = current_runtime - prev_runtime
        trends["runtime_change"] = f"{change:+.1f}s"
    elif current_runtime > 0:
        trends["runtime_change"] = f"+{current_runtime:.1f}s"  # First run

    # Coverage change
    current_coverage = current.get("metrics", {}).get("coverage", {}).get("overall", 0)
    prev_coverage = previous.get("metrics", {}).get("coverage", {}).get("overall", 0)
    if prev_coverage > 0:
        change = current_coverage - prev_coverage
        trends["coverage_change"] = f"{change:+.1f}%"
    elif current_coverage > 0:
        trends["coverage_change"] = f"+{current_coverage:.1f}%"  # First run

    # Test count change
    current_total = current.get("metrics", {}).get("test_health", {}).get("total", 0)
    prev_total = previous.get("metrics", {}).get("test_health", {}).get("total", 0)
    trends["test_count_change"] = f"{current_total - prev_total:+d}"

    # Pipeline metrics trends
    current_pipeline = current.get("metrics", {}).get("pipeline", {})
    prev_pipeline = previous.get("metrics", {}).get("pipeline", {})
    if current_pipeline and prev_pipeline:
        # Pipeline run duration change
        current_duration = current_pipeline.get("run_duration_seconds", 0)
        prev_duration = prev_pipeline.get("run_duration_seconds", 0)
        if prev_duration > 0:
            change = current_duration - prev_duration
            trends["pipeline_duration_change"] = f"{change:+.1f}s"
        elif current_duration > 0:
            trends["pipeline_duration_change"] = f"+{current_duration:.1f}s"  # First run

        # Episodes scraped change
        current_episodes = current_pipeline.get("episodes_scraped_total", 0)
        prev_episodes = prev_pipeline.get("episodes_scraped_total", 0)
        trends["pipeline_episodes_change"] = f"{current_episodes - prev_episodes:+d}"

        # Transcripts transcribed change
        current_transcribed = current_pipeline.get("transcripts_transcribed", 0)
        prev_transcribed = prev_pipeline.get("transcripts_transcribed", 0)
        trends["pipeline_transcribed_change"] = f"{current_transcribed - prev_transcribed:+d}"

    return trends


def detect_deviations(
    current: Dict[str, Any], history: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Detect metric deviations compared to historical data."""
    alerts = []

    if len(history) < 2:
        return alerts

    # Get current metrics
    current_runtime = current.get("metrics", {}).get("runtime", {}).get("total", 0)
    current_coverage = current.get("metrics", {}).get("coverage", {}).get("overall", 0)
    current_total = current.get("metrics", {}).get("test_health", {}).get("total", 0)
    current_flaky = current.get("metrics", {}).get("test_health", {}).get("flaky", 0)
    current_complexity = (
        current.get("metrics", {}).get("complexity", {}).get("cyclomatic_complexity", 0)
    )
    current_mi = current.get("metrics", {}).get("complexity", {}).get("maintainability_index", 0)

    # Runtime deviation (compare with last 5 runs)
    recent_runtimes = [
        h.get("metrics", {}).get("runtime", {}).get("total", 0)
        for h in history[-5:]
        if h.get("metrics", {}).get("runtime", {}).get("total", 0) > 0
    ]
    if recent_runtimes and current_runtime > 0:
        sorted_runtimes = sorted(recent_runtimes)
        median_runtime = sorted_runtimes[len(sorted_runtimes) // 2]
        if median_runtime > 0 and current_runtime > median_runtime * 1.1:
            pct_change = ((current_runtime / median_runtime) - 1) * 100
            severity = "warning" if pct_change < 20 else "error"
            alerts.append(
                {
                    "type": "regression",
                    "metric": "runtime",
                    "severity": severity,
                    "message": (
                        f"Runtime increased by {pct_change:.1f}% compared to "
                        f"last {len(recent_runtimes)} runs"
                    ),
                }
            )

    # Coverage drop (compare with last 10 runs)
    recent_coverages = [
        h.get("metrics", {}).get("coverage", {}).get("overall", 0)
        for h in history[-10:]
        if h.get("metrics", {}).get("coverage", {}).get("overall", 0) > 0
    ]
    if recent_coverages and current_coverage > 0:
        avg_coverage = sum(recent_coverages) / len(recent_coverages)
        if current_coverage < avg_coverage - 1.0:
            alerts.append(
                {
                    "type": "regression",
                    "metric": "coverage",
                    "severity": "error",
                    "message": (
                        f"Coverage dropped by {avg_coverage - current_coverage:.1f}% "
                        f"(avg: {avg_coverage:.1f}%)"
                    ),
                }
            )

    # Test count change (alert if significant change)
    recent_totals = [
        h.get("metrics", {}).get("test_health", {}).get("total", 0) for h in history[-5:]
    ]
    if recent_totals:
        avg_total = sum(recent_totals) / len(recent_totals)
        if abs(current_total - avg_total) > avg_total * 0.05:  # > 5% change
            change = current_total - avg_total
            alerts.append(
                {
                    "type": "change",
                    "metric": "test_count",
                    "severity": "info",
                    "message": f"Test count changed by {change:+d} (avg: {avg_total:.0f})",
                }
            )

    # Flaky test increase
    recent_flaky = [
        h.get("metrics", {}).get("test_health", {}).get("flaky", 0) for h in history[-5:]
    ]
    if recent_flaky and current_flaky > 0:
        max_recent_flaky = max(recent_flaky) if recent_flaky else 0
        if max_recent_flaky > 0 and current_flaky > max_recent_flaky * 1.5:
            alerts.append(
                {
                    "type": "regression",
                    "metric": "flaky_tests",
                    "severity": "warning",
                    "message": (
                        f"Flaky test count increased to {current_flaky} "
                        f"(was {max_recent_flaky})"
                    ),
                }
            )
        elif max_recent_flaky == 0 and current_flaky > 0:
            alerts.append(
                {
                    "type": "regression",
                    "metric": "flaky_tests",
                    "severity": "warning",
                    "message": f"Flaky tests detected: {current_flaky} (previously 0)",
                }
            )

    # Complexity increase (compare with last 10 runs)
    recent_complexities = [
        h.get("metrics", {}).get("complexity", {}).get("cyclomatic_complexity", 0)
        for h in history[-10:]
        if h.get("metrics", {}).get("complexity", {}).get("cyclomatic_complexity", 0) > 0
    ]
    if recent_complexities and current_complexity > 0:
        avg_complexity = sum(recent_complexities) / len(recent_complexities)
        if current_complexity > avg_complexity * 1.1:  # >10% increase
            pct_change = ((current_complexity / avg_complexity) - 1) * 100
            severity = "warning" if pct_change < 20 else "error"
            alerts.append(
                {
                    "type": "regression",
                    "metric": "complexity",
                    "severity": severity,
                    "message": (
                        f"Complexity increased by {pct_change:.1f}% compared to "
                        f"last {len(recent_complexities)} runs (avg: {avg_complexity:.1f})"
                    ),
                }
            )

    # Maintainability drop (compare with last 10 runs)
    recent_mi = [
        h.get("metrics", {}).get("complexity", {}).get("maintainability_index", 0)
        for h in history[-10:]
        if h.get("metrics", {}).get("complexity", {}).get("maintainability_index", 0) > 0
    ]
    if recent_mi and current_mi > 0:
        avg_mi = sum(recent_mi) / len(recent_mi)
        if current_mi < avg_mi - 2.0:  # Drop of 2+ points
            alerts.append(
                {
                    "type": "regression",
                    "metric": "maintainability",
                    "severity": "warning",
                    "message": (
                        f"Maintainability dropped by {avg_mi - current_mi:.1f} points "
                        f"(avg: {avg_mi:.1f}, current: {current_mi:.1f})"
                    ),
                }
            )

    # File-level complexity (from wily trends)
    files_degrading = current.get("metrics", {}).get("complexity", {}).get("files_degrading", [])
    if files_degrading:
        alerts.append(
            {
                "type": "regression",
                "metric": "file_complexity",
                "severity": "info",
                "message": (
                    f"Files with increasing complexity: "
                    f"{', '.join(files_degrading[:5])}"
                    + ("..." if len(files_degrading) > 5 else "")
                ),
            }
        )

    return alerts


def extract_pipeline_metrics(pipeline_metrics_path: Path) -> dict:
    """Extract pipeline performance metrics from JSON file.

    Args:
        pipeline_metrics_path: Path to pipeline metrics JSON file

    Returns:
        Dictionary with pipeline metrics or empty dict if file not found
    """
    if not pipeline_metrics_path.exists():
        return {}

    try:
        with open(pipeline_metrics_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load pipeline metrics from {pipeline_metrics_path}: {e}")
        return {}


def generate_metrics(
    reports_dir: Path,
    output_path: Path,
    history_path: Optional[Path] = None,
    commit: str = None,
    branch: str = None,
    workflow_run_url: str = None,
    pipeline_metrics_path: Optional[Path] = None,
    coverage_threshold: float = 80.0,
) -> None:
    """Generate metrics JSON from test artifacts.

    Args:
        reports_dir: Directory containing test artifacts
        output_path: Path to output metrics JSON file
        history_path: Optional path to history.jsonl file for trend calculation
        commit: Git commit SHA (default: GITHUB_SHA env var)
        branch: Git branch/ref (default: GITHUB_REF env var)
        workflow_run_url: Workflow run URL (default: constructed from env vars)
        pipeline_metrics_path: Optional path to pipeline metrics JSON file
        coverage_threshold: Coverage threshold for combined coverage (default: 75.0)
    """

    pytest_json_path = reports_dir / "pytest.json"
    coverage_xml_path = reports_dir / "coverage.xml"

    # Get environment variables if not provided
    if commit is None:
        commit = os.environ.get("GITHUB_SHA", "unknown")
    if branch is None:
        branch = os.environ.get("GITHUB_REF", "unknown")
    if workflow_run_url is None:
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        run_id = os.environ.get("GITHUB_RUN_ID", "")
        if repo and run_id:
            workflow_run_url = f"https://github.com/{repo}/actions/runs/{run_id}"
        else:
            workflow_run_url = ""

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "commit": commit,
        "branch": branch,
        "workflow_run": workflow_run_url,
        "metrics": {
            "runtime": extract_runtime_metrics(pytest_json_path),
            "test_health": extract_test_metrics(pytest_json_path),
            "coverage": extract_coverage_metrics(coverage_xml_path, threshold=coverage_threshold),
            "slowest_tests": extract_slowest_tests(reports_dir),
            "complexity": extract_complexity_metrics(reports_dir),
        },
    }

    # Add pipeline metrics if available
    if pipeline_metrics_path:
        pipeline_metrics = extract_pipeline_metrics(pipeline_metrics_path)
        if pipeline_metrics:
            metrics["metrics"]["pipeline"] = pipeline_metrics

    # Load history and calculate trends/alerts
    history = []
    previous = None
    if history_path and history_path.exists():
        history = load_history(history_path)
        if history:
            previous = history[-1]  # Most recent

    # Calculate trends (compare with previous run)
    if previous:
        metrics["trends"] = calculate_trends(metrics, previous)
    else:
        metrics["trends"] = {}

    # Detect deviations (compare with history)
    if len(history) >= 2:
        metrics["alerts"] = detect_deviations(metrics, history)
    else:
        metrics["alerts"] = []

    # Write metrics JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Generated metrics JSON: {output_path}")
    print(f"   - Total tests: {metrics['metrics']['test_health'].get('total', 0)}")
    print(f"   - Coverage: {metrics['metrics']['coverage'].get('overall', 0):.1f}%")
    print(f"   - Runtime: {metrics['metrics']['runtime'].get('total', 0):.1f}s")

    # Print trends if available
    if metrics.get("trends"):
        trends = metrics["trends"]
        if trends.get("runtime_change"):
            print(f"   - Runtime change: {trends['runtime_change']}")
        if trends.get("coverage_change"):
            print(f"   - Coverage change: {trends['coverage_change']}")
        if trends.get("test_count_change"):
            print(f"   - Test count change: {trends['test_count_change']}")

    # Print alerts if any
    if metrics.get("alerts"):
        print(f"   - ⚠️  {len(metrics['alerts'])} alert(s) detected:")
        for alert in metrics["alerts"]:
            severity_icon = (
                "🔴"
                if alert["severity"] == "error"
                else "⚠️" if alert["severity"] == "warning" else "ℹ️"
            )
            print(f"     {severity_icon} {alert['message']}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate metrics JSON from test artifacts")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing test artifacts (default: reports)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/latest.json"),
        help="Output path for metrics JSON (default: metrics/latest.json)",
    )
    parser.add_argument(
        "--commit",
        help="Git commit SHA (default: GITHUB_SHA env var)",
    )
    parser.add_argument(
        "--branch",
        help="Git branch/ref (default: GITHUB_REF env var)",
    )
    parser.add_argument(
        "--workflow-run",
        help="Workflow run URL (default: constructed from GITHUB env vars)",
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="Path to history.jsonl file for trend and alert calculation (optional)",
    )
    parser.add_argument(
        "--pipeline-metrics",
        type=Path,
        help="Path to pipeline metrics JSON file (optional)",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Coverage threshold percentage for combined coverage (default: 80.0). "
        "Note: This is only enforced on combined coverage, not on individual test types.",
    )

    args = parser.parse_args()

    try:
        generate_metrics(
            reports_dir=args.reports_dir,
            output_path=args.output,
            history_path=args.history,
            commit=args.commit,
            branch=args.branch,
            workflow_run_url=args.workflow_run,
            pipeline_metrics_path=args.pipeline_metrics,
            coverage_threshold=args.coverage_threshold,
        )
    except Exception as e:
        print(f"❌ Error generating metrics: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
