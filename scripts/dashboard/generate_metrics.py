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


def _json_duration_value(raw: Any) -> float:
    """Coerce pytest-json-report duration (number or numeric string) to float."""
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and raw != raw:  # NaN
            return 0.0
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except ValueError:
            return 0.0
    return 0.0


def pytest_json_test_duration_seconds(test: Dict[str, Any]) -> float:
    """Wall time for one test from pytest-json-report.

    The plugin usually omits a top-level ``duration``; use setup/call/teardown
    stage durations when needed.
    """
    raw = test.get("duration")
    top = _json_duration_value(raw) if raw is not None else 0.0
    if top > 0:
        return top
    total = 0.0
    for when in ("setup", "call", "teardown"):
        stage = test.get(when)
        if isinstance(stage, dict):
            total += _json_duration_value(stage.get("duration"))
    return total


def pytest_json_test_passed_after_rerun(test: Dict[str, Any]) -> bool:
    """Return True if this test passed only after pytest-rerunfailures retried it.

    ``pytest-json-report`` does **not** set a top-level ``rerun: true`` flag. For a
    test that fails then passes, the report keeps top-level ``outcome`` as
    ``rerun`` while ``call.outcome`` is ``passed``. A clean first-pass test has
    top-level ``outcome`` ``passed``.

    We also accept ``outcome == passed`` with ``rerun is True`` for compatibility
    if a future report format adds that field.
    """
    if test.get("outcome") == "passed" and test.get("rerun") is True:
        return True
    call = test.get("call") or {}
    return test.get("outcome") == "rerun" and call.get("outcome") == "passed"


def _merge_pytest_test_record_for_health(
    prev: Dict[str, Any], new: Dict[str, Any]
) -> Dict[str, Any]:
    """When the same nodeid appears in merged + shard JSON, keep flaky-consistent data.

    Merged ``pytest.json`` can lose ``outcome: rerun`` while per-job shards still
    have it; merging must preserve a passed-after-rerun signal from any source.
    """
    p_flaky = pytest_json_test_passed_after_rerun(prev)
    n_flaky = pytest_json_test_passed_after_rerun(new)
    if n_flaky and not p_flaky:
        return new
    if p_flaky and not n_flaky:
        return prev
    return new


def _build_test_health_metrics(tests: List[Dict[str, Any]], summary: Dict[str, Any]) -> dict:
    """Shared test_health dict from a test list + summary block."""
    flaky_tests = [t for t in tests if pytest_json_test_passed_after_rerun(t)]
    flaky_count = len(flaky_tests)

    flaky_details = []
    for test in flaky_tests:
        duration = test.get("duration")
        if duration is None:
            call = test.get("call") or {}
            duration = call.get("duration", 0)
        flaky_details.append(
            {
                "name": test.get("nodeid", "unknown"),
                "duration": duration,
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


def extract_test_metrics(pytest_json_path: Path) -> dict:
    """Extract test health metrics from a single pytest JSON report."""
    if not pytest_json_path.exists():
        return {}

    with open(pytest_json_path) as f:
        pytest_data = json.load(f)

    return _build_test_health_metrics(
        pytest_data.get("tests", []),
        pytest_data.get("summary", {}),
    )


def extract_test_metrics_from_reports_dir(reports_dir: Path) -> dict:
    """Extract test health from all pytest JSON under ``reports_dir`` (CI / nightly).

    Combines ``pytest-*.json`` shards and ``pytest.json`` so flaky tests are not
    dropped when duplicate ``nodeid`` rows disagree or merged output omits rerun
    metadata. Summary totals prefer ``pytest.json`` when its ``summary.total`` is
    positive; otherwise sums per-shard summaries.
    """
    paths: List[Path] = []
    for p in sorted(reports_dir.glob("pytest-*.json")):
        if p.is_file():
            paths.append(p)
    merged_path = reports_dir / "pytest.json"
    if merged_path.is_file():
        paths.append(merged_path)

    by_node: Dict[str, Dict[str, Any]] = {}
    shard_summary = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
    merged_summary: Optional[Dict[str, Any]] = None

    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping pytest JSON %s: %s", path, e)
            continue
        if path.name == "pytest.json":
            merged_summary = data.get("summary") or {}
        else:
            s = data.get("summary") or {}
            for k in shard_summary:
                shard_summary[k] += int(s.get(k, 0) or 0)
        for t in data.get("tests", []):
            nid = (t.get("nodeid") or "").strip()
            if not nid:
                continue
            prev = by_node.get(nid)
            by_node[nid] = t if prev is None else _merge_pytest_test_record_for_health(prev, t)

    tests = list(by_node.values())
    if merged_summary and int(merged_summary.get("total", 0) or 0) > 0:
        summary_use: Dict[str, Any] = merged_summary
    else:
        summary_use = shard_summary

    return _build_test_health_metrics(tests, summary_use)


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
            duration = pytest_json_test_duration_seconds(test)
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


def _dedupe_slowest_by_name_max_duration(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one row per test name with the largest duration (merge shards)."""
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = row.get("name", "unknown")
        prev = best.get(name)
        if prev is None or row.get("duration", 0) > prev.get("duration", 0):
            best[name] = row
    return list(best.values())


# Per-job reports (uploaded as separate artifacts). Prefer these for slowest tests:
# the merged ``pytest.json`` in CI can lack usable per-test timings in some cases.
_PYTEST_SHARD_JSON = (
    "pytest-unit.json",
    "pytest-integration.json",
    "pytest-e2e.json",
    "pytest-e2e-serial.json",
    "pytest-nightly.json",
)


def extract_slowest_tests(reports_dir: Path, top_n: int = 10) -> list:
    """Extract slowest tests from pytest JSON reports and JUnit XML.

    Collects timed tests from pytest JSON (merged + per-job shards when present).
    **Always** also reads every ``junit*.xml`` under ``reports_dir`` and merges via
    :func:`_dedupe_slowest_by_name_max_duration` (max duration wins per name).

    Without this merge, sparse JSON (e.g. only a few tests with non-zero durations
    under xdist) would cap ``slowest_tests`` at that small count and **skip** JUnit
    entirely—the old logic only opened JUnit when JSON yielded zero timed tests.

    Args:
        reports_dir: Directory containing test reports
        top_n: Number of slowest tests to return (default: 10; matches dashboard table)

    Returns:
        List of dicts with 'name' and 'duration' keys, sorted by duration descending
    """
    tests: List[Dict[str, Any]] = []

    shard_paths = [reports_dir / p for p in _PYTEST_SHARD_JSON if (reports_dir / p).is_file()]
    merged_path = reports_dir / "pytest.json"

    paths_to_read: List[Path] = []
    if shard_paths:
        paths_to_read = shard_paths
    elif merged_path.is_file():
        paths_to_read = [merged_path]

    for json_path in paths_to_read:
        extracted = _extract_tests_from_json(json_path)
        tests.extend(extracted)
        logger.debug("Extracted %d timed tests from %s", len(extracted), json_path.name)

    tests = _dedupe_slowest_by_name_max_duration(tests)

    # Shards present but no positive durations — try merged blob as fallback.
    if not tests and shard_paths and merged_path.is_file():
        tests = _extract_tests_from_json(merged_path)
        tests = _dedupe_slowest_by_name_max_duration(tests)
        logger.debug(
            "Slowest: shard files had no durations; merged pytest.json yielded %d rows",
            len(tests),
        )

    if not tests and not paths_to_read:
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
                logger.debug("Extracted %d tests from %s", len(extracted), pattern)
        tests = _dedupe_slowest_by_name_max_duration(tests)

    junit_tests: List[Dict[str, Any]] = []
    for junit_xml_path in sorted(reports_dir.glob("junit*.xml")):
        extracted = _extract_tests_from_junit(junit_xml_path)
        junit_tests.extend(extracted)
        logger.debug(
            "Slowest: extracted %d timed tests from %s",
            len(extracted),
            junit_xml_path.name,
        )

    combined = _dedupe_slowest_by_name_max_duration(tests + junit_tests)
    combined.sort(key=lambda x: x["duration"], reverse=True)
    return combined[:top_n]


def load_history(history_path: Path) -> List[Dict[str, Any]]:
    """Load historical metrics from history.jsonl (tolerates legacy multi-line appends)."""
    dash_dir = Path(__file__).resolve().parent
    if str(dash_dir) not in sys.path:
        sys.path.insert(0, str(dash_dir))
    from metrics_jsonl import load_metrics_history

    return load_metrics_history(history_path)


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
    trends["test_count_change"] = f"{int(current_total - prev_total):+d}"

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
        trends["pipeline_episodes_change"] = f"{int(current_episodes - prev_episodes):+d}"

        # Transcripts transcribed change
        current_transcribed = current_pipeline.get("transcripts_transcribed", 0)
        prev_transcribed = prev_pipeline.get("transcripts_transcribed", 0)
        trends["pipeline_transcribed_change"] = f"{int(current_transcribed - prev_transcribed):+d}"

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
                    "message": f"Test count changed by {int(change):+d} (avg: {avg_total:.0f})",
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
    coverage_threshold: float = 70.0,
    slowest_top_n: int = 10,
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
        coverage_threshold: Coverage threshold for combined coverage (default: 70.0)
        slowest_top_n: Max slowest-test rows in output (default: 10; same as dashboard)
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
            "test_health": extract_test_metrics_from_reports_dir(reports_dir),
            "coverage": extract_coverage_metrics(coverage_xml_path, threshold=coverage_threshold),
            "slowest_tests": extract_slowest_tests(reports_dir, top_n=slowest_top_n),
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
    print(f"   - Flaky (passed on rerun): {metrics['metrics']['test_health'].get('flaky', 0)}")
    print(f"   - Slowest tests (in JSON): {len(metrics['metrics'].get('slowest_tests') or [])}")
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
        default=70.0,
        help="Coverage threshold percentage for combined coverage (default: 70.0). "
        "Note: This is only enforced on combined coverage, not on individual test types.",
    )
    parser.add_argument(
        "--slowest-top-n",
        type=int,
        default=10,
        metavar="N",
        help="Number of slowest tests to include in metrics JSON (default: 10)",
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
            slowest_top_n=args.slowest_top_n,
        )
    except Exception as e:
        print(f"❌ Error generating metrics: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
