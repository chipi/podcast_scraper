#!/usr/bin/env python3
"""
Generate HTML dashboard for test metrics visualization.

This script generates an interactive HTML dashboard that displays:
- Current metrics (latest run)
- Trend charts (last 30 runs)
- Deviation alerts
- Slowest tests
- Coverage trends

It supports a --unified mode which includes a dropdown to switch between
CI and Nightly data sources.

See RFC-026: Metrics Consumption and Dashboards
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Must match ``generate_metrics.py`` default ``--slowest-top-n`` (rows stored in latest-*.json).
SLOWEST_TESTS_TABLE_MAX = 10


def load_latest_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
    """Load latest metrics from JSON file."""
    if not metrics_path.exists():
        print(f"⚠️ Warning: Metrics file not found: {metrics_path}", file=sys.stderr)
        return None

    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Warning: Could not load metrics: {e}", file=sys.stderr)
        return None


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


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ts


def get_severity_class(severity: str) -> str:
    """Get CSS class for alert severity."""
    severity_map = {
        "error": "alert-error",
        "warning": "alert-warning",
        "info": "alert-info",
    }
    return severity_map.get(severity, "alert-info")


def generate_static_dashboard(
    metrics_path: Path,
    history_path: Optional[Path],
    output_path: Path,
) -> None:
    """Generate a static HTML dashboard with baked-in data."""
    # Load data
    latest = load_latest_metrics(metrics_path)
    if not latest:
        print(f"❌ Error: Could not load metrics from {metrics_path}", file=sys.stderr)
        sys.exit(1)

    history = []
    if history_path and history_path.exists():
        history = load_history(history_path)
        # Include latest in history for charts (if not already there)
        if history and history[-1].get("timestamp") != latest.get("timestamp"):
            history.append(latest)
        elif not history:
            history = [latest]
    else:
        history = [latest]

    # Get metrics
    metrics = latest.get("metrics", {})
    runtime = metrics.get("runtime", {})
    test_health = metrics.get("test_health", {})
    coverage = metrics.get("coverage", {})
    complexity = metrics.get("complexity", {})
    pipeline_metrics = metrics.get("pipeline", {})
    slowest_tests = metrics.get("slowest_tests", [])
    flaky_tests = test_health.get("flaky_tests", [])
    trends = latest.get("trends", {})
    alerts = latest.get("alerts", [])

    # Prepare chart data (last 30 runs)
    chart_history = history[-30:] if len(history) > 30 else history
    runtime_data = [h.get("metrics", {}).get("runtime", {}).get("total", 0) for h in chart_history]
    coverage_data = [
        h.get("metrics", {}).get("coverage", {}).get("overall", 0) for h in chart_history
    ]
    test_count_data = [
        h.get("metrics", {}).get("test_health", {}).get("total", 0) for h in chart_history
    ]
    flaky_data = [
        h.get("metrics", {}).get("test_health", {}).get("flaky", 0) for h in chart_history
    ]
    complexity_data = [
        h.get("metrics", {}).get("complexity", {}).get("cyclomatic_complexity", 0)
        for h in chart_history
    ]
    maintainability_data = [
        h.get("metrics", {}).get("complexity", {}).get("maintainability_index", 0)
        for h in chart_history
    ]
    docstrings_data = [
        h.get("metrics", {}).get("complexity", {}).get("docstring_coverage", 0)
        for h in chart_history
    ]
    dead_code_data = [
        h.get("metrics", {}).get("complexity", {}).get("dead_code_count", 0) for h in chart_history
    ]
    spelling_errors_data = [
        h.get("metrics", {}).get("complexity", {}).get("spelling_errors_count", 0)
        for h in chart_history
    ]
    pipeline_run_duration_data = [
        h.get("metrics", {}).get("pipeline", {}).get("run_duration_seconds", 0)
        for h in chart_history
    ]
    pipeline_episodes_scraped_data = [
        h.get("metrics", {}).get("pipeline", {}).get("episodes_scraped_total", 0)
        for h in chart_history
    ]
    timestamps = [h.get("timestamp", "")[:10] for h in chart_history]  # Just the date part

    # Build HTML components
    test_count_trend = ""
    if trends.get("test_count_change"):
        test_count_trend = f'<div class="metric-trend">{trends.get("test_count_change")}</div>'

    runtime_trend = ""
    if trends.get("runtime_change"):
        runtime_change = trends.get("runtime_change", "")
        trend_class = (
            "up"
            if runtime_change.startswith("+")
            else "down" if runtime_change.startswith("-") else "neutral"
        )
        runtime_trend = f'<div class="metric-trend trend-{trend_class}">{runtime_change}</div>'

    coverage_trend = ""
    if trends.get("coverage_change"):
        coverage_change = trends.get("coverage_change", "")
        trend_class = (
            "up"
            if coverage_change.startswith("+")
            else "down" if coverage_change.startswith("-") else "neutral"
        )
        coverage_trend = f'<div class="metric-trend trend-{trend_class}">{coverage_change}</div>'

    # Pipeline metrics trends
    pipeline_duration_trend = ""
    if trends.get("pipeline_duration_change"):
        duration_change = trends.get("pipeline_duration_change", "")
        trend_class = (
            "up"
            if duration_change.startswith("+")
            else "down" if duration_change.startswith("-") else "neutral"
        )
        pipeline_duration_trend = (
            f'<div class="metric-trend trend-{trend_class}">{duration_change}</div>'
        )

    pipeline_episodes_trend = ""
    if trends.get("pipeline_episodes_change"):
        episodes_change = trends.get("pipeline_episodes_change", "")
        trend_class = (
            "up"
            if episodes_change.startswith("+")
            else "down" if episodes_change.startswith("-") else "neutral"
        )
        pipeline_episodes_trend = (
            f'<div class="metric-trend trend-{trend_class}">{episodes_change}</div>'
        )

    # Code quality trends (from wily / metrics history)
    complexity_trend_val = complexity.get("complexity_trend", "N/A")
    complexity_trend_html = ""
    if complexity_trend_val and complexity_trend_val != "N/A":
        trend_class = (
            "up"
            if complexity_trend_val.startswith("+")
            else "down" if complexity_trend_val.startswith("-") else "neutral"
        )
        complexity_trend_html = (
            f'<div class="metric-trend trend-{trend_class}">{complexity_trend_val}</div>'
        )
    maintainability_trend_val = complexity.get("maintainability_trend", "N/A")
    maintainability_trend_html = ""
    if maintainability_trend_val and maintainability_trend_val != "N/A":
        trend_class = (
            "up"
            if maintainability_trend_val.startswith("+")
            else "down" if maintainability_trend_val.startswith("-") else "neutral"
        )
        maintainability_trend_html = (
            f'<div class="metric-trend trend-{trend_class}">{maintainability_trend_val}</div>'
        )

    files_degrading = complexity.get("files_degrading", []) or []
    files_improving = complexity.get("files_improving", []) or []
    files_degrading_html = ""
    if files_degrading:
        lst = ", ".join(files_degrading[:8]) + ("..." if len(files_degrading) > 8 else "")
        files_degrading_html = (
            f'<div class="metric-card" style="grid-column: 1 / -1;">'
            f'<div class="metric-label">Files with increasing complexity</div>'
            f'<div class="metric-value" style="font-size: 0.9em;">{lst}</div></div>'
        )
    files_improving_html = ""
    if files_improving:
        lst = ", ".join(files_improving[:8]) + ("..." if len(files_improving) > 8 else "")
        files_improving_html = (
            f'<div class="metric-card" style="grid-column: 1 / -1;">'
            f'<div class="metric-label">Files with improving complexity</div>'
            f'<div class="metric-value" style="font-size: 0.9em;">{lst}</div></div>'
        )

    flaky_color = "#e74c3c" if test_health.get("flaky", 0) > 0 else "#27ae60"

    # Build pipeline metrics HTML
    pipeline_metrics_html = ""
    if pipeline_metrics:
        run_duration = pipeline_metrics.get("run_duration_seconds", 0)
        episodes_scraped = pipeline_metrics.get("episodes_scraped_total", 0)
        episodes_skipped = pipeline_metrics.get("episodes_skipped_total", 0)
        transcripts_downloaded = pipeline_metrics.get("transcripts_downloaded", 0)
        transcripts_transcribed = pipeline_metrics.get("transcripts_transcribed", 0)
        episodes_summarized = pipeline_metrics.get("episodes_summarized", 0)
        metadata_files = pipeline_metrics.get("metadata_files_generated", 0)

        pipeline_metrics_html = f"""
            <div class="metric-card">
                <div class="metric-label">Pipeline Run Duration</div>
                <div class="metric-value">{run_duration:.1f}s</div>
                {pipeline_duration_trend}
            </div>
            <div class="metric-card">
                <div class="metric-label">Episodes Scraped</div>
                <div class="metric-value">{episodes_scraped}</div>
                {pipeline_episodes_trend}
            </div>
            <div class="metric-card">
                <div class="metric-label">Episodes Skipped</div>
                <div class="metric-value">{episodes_skipped}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Transcripts Downloaded</div>
                <div class="metric-value">{transcripts_downloaded}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Transcripts Transcribed</div>
                <div class="metric-value">{transcripts_transcribed}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Episodes Summarized</div>
                <div class="metric-value">{episodes_summarized}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Metadata Files</div>
                <div class="metric-value">{metadata_files}</div>
            </div>
        """

    # Build pipeline performance section HTML and chart code
    pipeline_performance_section = ""
    pipeline_chart_code = ""
    if pipeline_metrics:
        pipeline_performance_section = f"""
        <div class="chart-section">
            <h2>🚀 Pipeline Performance Trends (Last {len(chart_history)} Runs)</h2>
            <div class="chart-container">
                <canvas id="pipeline-chart"></canvas>
            </div>
        </div>
        """
        pipeline_chart_code = f"""
        const pipelineCtx = document.getElementById('pipeline-chart');
        if (pipelineCtx) {{
            new Chart(pipelineCtx.getContext('2d'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(timestamps)},
                    datasets: [
                        {{
                            label: 'Run Duration (s)',
                            data: {json.dumps(pipeline_run_duration_data)},
                            borderColor: 'rgb(155, 89, 182)',
                            backgroundColor: 'rgba(155, 89, 182, 0.1)',
                            yAxisID: 'y',
                            tension: 0.1
                        }},
                        {{
                            label: 'Episodes Scraped',
                            data: {json.dumps(pipeline_episodes_scraped_data)},
                            borderColor: 'rgb(26, 188, 156)',
                            backgroundColor: 'rgba(26, 188, 156, 0.1)',
                            yAxisID: 'y1',
                            tension: 0.1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false,
                    }},
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'top',
                        }},
                        tooltip: {{
                            enabled: true,
                        }}
                    }},
                    scales: {{
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {{
                                display: true,
                                text: 'Run Duration (seconds)'
                            }}
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {{
                                display: true,
                                text: 'Episodes Scraped'
                            }},
                            grid: {{
                                drawOnChartArea: false,
                            }}
                        }}
                    }}
                }}
            }});
        }}
        """

    alerts_html = ""
    if alerts:
        alerts_list = []
        for alert in alerts:
            severity_class = get_severity_class(alert.get("severity", "info"))
            metric = alert.get("metric", "unknown").upper()
            message = alert.get("message", "")
            alerts_list.append(
                f'<div class="alert {severity_class}"><strong>{metric}</strong>: {message}</div>'
            )
        alerts_html = "\n            ".join(alerts_list)
    else:
        alerts_html = '<div class="no-data">✅ No alerts - all metrics within normal range</div>'

    slowest_tests_html = ""
    if slowest_tests:
        test_rows = []
        for test in slowest_tests[:SLOWEST_TESTS_TABLE_MAX]:
            test_name = test.get("name", "unknown")
            duration = test.get("duration", 0)
            test_rows.append(f"<tr><td><code>{test_name}</code></td><td>{duration:.2f}s</td></tr>")
        slowest_tests_html = f"""
            <table>
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(test_rows)}
                </tbody>
            </table>
        """
    else:
        slowest_tests_html = '<div class="no-data">No test duration data available</div>'

    flaky_tests_html = ""
    if flaky_tests:
        flaky_rows = []
        for test in flaky_tests:
            test_name = test.get("name", "unknown")
            duration = test.get("duration", 0)
            flaky_rows.append(f"<tr><td><code>{test_name}</code></td><td>{duration:.2f}s</td></tr>")
        flaky_tests_html = f"""
            <table>
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(flaky_rows)}
                </tbody>
            </table>
        """
    else:
        flaky_tests_html = (
            '<div class="no-data">✅ No tests passed after a rerun in merged pytest JSON.</div>'
        )

    # Common head and CSS
    head_html = """
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Metrics Dashboard - Podcast Scraper</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            display: flex;
            flex-direction: row;
            gap: 40px;
            align-items: flex-start;
        }

        .header-title {
            flex: 1;
        }

        .header-alerts {
            flex: 1;
        }

        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .metric-label {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }

        .metric-trend {
            font-size: 0.9em;
            margin-top: 8px;
        }

        .trend-up {
            color: #27ae60;
        }

        .trend-down {
            color: #e74c3c;
        }

        .trend-neutral {
            color: #7f8c8d;
        }

        .alert {
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
            border-left: 4px solid;
        }

        .alert-error {
            background: #fee;
            border-color: #e74c3c;
            color: #c0392b;
        }

        .alert-warning {
            background: #fffbf0;
            border-color: #f39c12;
            color: #d68910;
        }

        .alert-info {
            background: #ebf5fb;
            border-color: #3498db;
            color: #2874a6;
        }

        .chart-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }

        .slowest-tests {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }

        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }

        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }

        .no-data {
            text-align: center;
            color: #7f8c8d;
            padding: 40px;
        }

        .source-selector {
            margin-bottom: 20px;
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 15px;
        }

        select {
            padding: 8px 12px;
            border-radius: 4px;
            border: 1px solid #ddd;
            font-size: 1em;
            color: #2c3e50;
            cursor: pointer;
        }

        .loader {
            display: none;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 2s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """

    body_html = f"""
    <div class="container">
        <header>
            <div class="header-title">
                <h1>📊 Test Metrics Dashboard</h1>
                <div class="timestamp">
                    Last updated: {format_timestamp(latest.get("timestamp", ""))}
                </div>
                <div class="timestamp">
                    Commit: <code>{latest.get("commit", "unknown")[:8]}</code> |
                    Branch: <code>{latest.get("branch", "unknown")}</code>
                </div>
            </div>
            <div class="header-alerts">
                <h1>⚠️ Alerts</h1>
                {alerts_html}
            </div>
        </header>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Tests</div>
                <div class="metric-value">{test_health.get("total", 0)}</div>
                {test_count_trend}
            </div>

            <div class="metric-card">
                <div class="metric-label">Passed</div>
                <div class="metric-value" style="color: #27ae60;">
                    {test_health.get("passed", 0)}
                </div>
                <div class="metric-trend">
                    Pass rate: {test_health.get("pass_rate", 0) * 100:.1f}%
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Failed</div>
                <div class="metric-value" style="color: #e74c3c;">
                    {test_health.get("failed", 0)}
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Skipped</div>
                <div class="metric-value" style="color: #f39c12;">
                    {test_health.get("skipped", 0)}
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Runtime</div>
                <div class="metric-value">{runtime.get("total", 0):.1f}s</div>
                {runtime_trend}
            </div>

            <div class="metric-card">
                <div class="metric-label">Coverage (threshold: {
                    coverage.get("threshold", 75):.0f}%)</div>
                <div class="metric-value" style="color: {
                    '#27ae60' if coverage.get('meets_threshold', True) else '#e74c3c'};">
                    {coverage.get("overall", 0):.1f}%
                </div>
                {coverage_trend}
            </div>

            <div class="metric-card">
                <div class="metric-label">Cyclomatic Complexity</div>
                <div class="metric-value">{complexity.get("cyclomatic_complexity", 0):.1f}</div>
                {complexity_trend_html}
            </div>

            <div class="metric-card">
                <div class="metric-label">Maintainability Index</div>
                <div class="metric-value">{complexity.get("maintainability_index", 0):.1f}</div>
                {maintainability_trend_html}
            </div>

            <div class="metric-card">
                <div class="metric-label">Docstring Coverage</div>
                <div class="metric-value">{complexity.get("docstring_coverage", 0):.1f}%</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Dead Code Items</div>
                <div class="metric-value" style="color: {
                    '#e74c3c' if complexity.get('dead_code_count', 0) > 0 else '#27ae60'};">
                    {complexity.get("dead_code_count", 0)}
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Spelling Errors</div>
                <div class="metric-value" style="color: {
                    '#e74c3c' if complexity.get('spelling_errors_count', 0) > 0 else '#27ae60'};">
                    {complexity.get("spelling_errors_count", 0)}
                </div>
            </div>
            {files_degrading_html}
            {files_improving_html}

            <div class="metric-card">
                <div class="metric-label">Flaky Tests</div>
                <div class="metric-value" style="color: {flaky_color};">
                    {test_health.get("flaky", 0)}
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Tests/Second</div>
                <div class="metric-value">{runtime.get("tests_per_second", 0):.1f}</div>
            </div>
            {pipeline_metrics_html}
        </div>

        <div class="chart-section">
            <h2>📈 Test Metrics Trends (Last {len(chart_history)} Runs)</h2>
            <div class="chart-container">
                <canvas id="trendsChart"></canvas>
            </div>
        </div>

        <div class="chart-section">
            <h2>📊 Code Quality Trends (Last {len(chart_history)} Runs)</h2>
            <div class="chart-container">
                <canvas id="quality-chart"></canvas>
            </div>
        </div>

        {pipeline_performance_section}

        <div class="slowest-tests">
            <h2>🐌 Slowest Tests (Top {SLOWEST_TESTS_TABLE_MAX})</h2>
            {slowest_tests_html}
        </div>

        <div class="slowest-tests">
            <h2>⚠️ Flaky Tests</h2>
            <p>Tests that failed initially but passed on rerun</p>
            {flaky_tests_html}
        </div>
    </div>
    """

    script_html = f"""
    <script>
        const timestamps = {json.dumps(timestamps)};
        const runtimeData = {json.dumps(runtime_data)};
        const coverageData = {json.dumps(coverage_data)};
        const testCountData = {json.dumps(test_count_data)};
        const complexityData = {json.dumps(complexity_data)};
        const maintainabilityData = {json.dumps(maintainability_data)};
        const docstringsData = {json.dumps(docstrings_data)};
        const deadCodeData = {json.dumps(dead_code_data)};
        const spellingErrorsData = {json.dumps(spelling_errors_data)};
        const flakyData = {json.dumps(flaky_data)};

        const trendsCtx = document.getElementById('trendsChart').getContext('2d');
        new Chart(trendsCtx, {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [
                    {{
                        label: 'Runtime (s)',
                        data: runtimeData,
                        borderColor: 'rgb(231, 76, 60)',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        yAxisID: 'y',
                        tension: 0.1
                    }},
                    {{
                        label: 'Coverage (%)',
                        data: coverageData,
                        borderColor: 'rgb(52, 152, 219)',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        yAxisID: 'y1',
                        tension: 0.1
                    }},
                    {{
                        label: 'Test Count',
                        data: testCountData,
                        borderColor: 'rgb(46, 204, 113)',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        yAxisID: 'y2',
                        tension: 0.1
                    }},
                    {{
                        label: 'Flaky Tests',
                        data: flakyData,
                        borderColor: 'rgb(243, 156, 18)',
                        backgroundColor: 'rgba(243, 156, 18, 0.1)',
                        yAxisID: 'y2',
                        tension: 0.1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top',
                    }},
                    tooltip: {{
                        enabled: true,
                    }}
                }},
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'Runtime (seconds)'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Coverage (%)'
                        }},
                        grid: {{
                            drawOnChartArea: false,
                        }}
                    }},
                    y2: {{
                        type: 'linear',
                        display: false,
                    }}
                }}
            }}
        }});

        // Code Quality Chart
        const qualityCtx = document.getElementById('quality-chart').getContext('2d');
        new Chart(qualityCtx, {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [
                    {{
                        label: 'Cyclomatic Complexity',
                        data: complexityData,
                        borderColor: 'rgb(231, 76, 60)',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        yAxisID: 'y',
                        tension: 0.1
                    }},
                    {{
                        label: 'Maintainability Index',
                        data: maintainabilityData,
                        borderColor: 'rgb(46, 204, 113)',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        yAxisID: 'y1',
                        tension: 0.1
                    }},
                    {{
                        label: 'Docstring Coverage (%)',
                        data: docstringsData,
                        borderColor: 'rgb(52, 152, 219)',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        yAxisID: 'y2',
                        tension: 0.1
                    }},
                    {{
                        label: 'Dead Code Items',
                        data: deadCodeData,
                        borderColor: 'rgb(231, 76, 60)',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        yAxisID: 'y3',
                        tension: 0.1
                    }},
                    {{
                        label: 'Spelling Errors',
                        data: spellingErrorsData,
                        borderColor: 'rgb(243, 156, 18)',
                        backgroundColor: 'rgba(243, 156, 18, 0.1)',
                        yAxisID: 'y3',
                        tension: 0.1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top',
                    }},
                    tooltip: {{
                        enabled: true,
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: 'Date'
                        }}
                    }},
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'Cyclomatic Complexity'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Maintainability Index'
                        }},
                        grid: {{
                            drawOnChartArea: false,
                        }}
                    }},
                    y2: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Docstring Coverage (%)'
                        }},
                        grid: {{
                            drawOnChartArea: false,
                        }}
                    }},
                    y3: {{
                        type: 'linear',
                        display: false,
                    }}
                }}
            }}
        }});

        {pipeline_chart_code}
    </script>
    """

    full_html = f'<!DOCTYPE html>\n<html lang="en">\n<head>\n{head_html}\n</head>\n<body>\n{body_html}\n{script_html}\n</body>\n</html>'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(full_html)

    print(f"✅ Generated static dashboard: {output_path}")


def generate_unified_dashboard(output_path: Path) -> None:
    """Generate a unified HTML dashboard that dynamically loads CI or Nightly data."""
    # This template is mostly JavaScript that fetches the JSON files.
    # We use the same CSS and base structure as the static dashboard.

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <title>Unified Test Metrics Dashboard - Podcast Scraper</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        /* Same CSS as static dashboard */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; display: flex; flex-direction: row; gap: 40px; align-items: flex-start; }
        .header-title { flex: 1; }
        .header-alerts { flex: 1; }
        h1 { color: #2c3e50; margin-bottom: 10px; }
        .timestamp { color: #7f8c8d; font-size: 0.9em; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-label { color: #7f8c8d; font-size: 0.9em; margin-bottom: 8px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .metric-trend { font-size: 0.9em; margin-top: 8px; }
        .trend-up { color: #27ae60; }
        .trend-down { color: #e74c3c; }
        .trend-neutral { color: #7f8c8d; }
        .alert { padding: 12px; border-radius: 4px; margin-bottom: 10px; border-left: 4px solid; }
        .alert-error { background: #fee; border-color: #e74c3c; color: #c0392b; }
        .alert-warning { background: #fffbf0; border-color: #f39c12; color: #d68910; }
        .alert-info { background: #ebf5fb; border-color: #3498db; color: #2874a6; }
        .chart-section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .chart-container { position: relative; height: 300px; margin-top: 20px; }
        .slowest-tests { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; color: #2c3e50; }
        .no-data { text-align: center; color: #7f8c8d; padding: 40px; }
        .chart-caption { font-size: 0.85em; color: #7f8c8d; margin: 0 0 12px 0; line-height: 1.45; }
        .chart-placeholder { padding: 24px 16px; min-height: 120px; display: flex; align-items: center; justify-content: center; }
        .source-selector { margin-bottom: 20px; background: white; padding: 15px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; align-items: center; gap: 15px; }
        select { padding: 8px 12px; border-radius: 4px; border: 1px solid #ddd; font-size: 1em; color: #2c3e50; cursor: pointer; }
        .loader { display: none; border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; width: 20px; height: 20px; animation: spin 2s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .data-mode-banner { margin-bottom: 16px; padding: 12px 16px; border-radius: 8px; font-size: 0.9em; line-height: 1.5; border: 1px solid #ccc; }
        .data-mode-loading { background: #f0f0f0; color: #555; }
        .data-mode-unified { background: #e8f6ef; border-color: #27ae60; color: #1b4332; }
        .data-mode-legacy { background: #fff9e6; border-color: #e6a23c; color: #5c4a00; }
        .data-mode-error { background: #fdecea; border-color: #e74c3c; color: #78281f; }
        .banner-hint { display: block; margin-top: 8px; font-size: 0.95em; font-weight: normal; }
    </style>
</head>
<body>
    <div class="container">
        <div class="source-selector">
            <label for="data-source"><strong>Data Source:</strong></label>
            <select id="data-source">
                <option value="ci">CI Metrics (Latest Push)</option>
                <option value="nightly">Nightly Metrics (Scheduled Runs)</option>
            </select>
            <div id="loader" class="loader"></div>
        </div>

        <div id="data-mode-banner" class="data-mode-banner data-mode-loading" role="status" aria-live="polite">Loading metrics…</div>

        <header id="dashboard-header">
            <div class="header-title">
                <h1>📊 Test Metrics Dashboard</h1>
                <div class="timestamp" id="last-updated">Loading...</div>
                <div class="timestamp" id="commit-info"></div>
                <div class="timestamp" id="workflow-link"></div>
            </div>
            <div class="header-alerts">
                <h1>⚠️ Alerts</h1>
                <div id="alerts-container">Loading alerts...</div>
            </div>
        </header>

        <div class="metrics-grid" id="metrics-grid">
            <!-- Metric cards will be populated here -->
        </div>

        <div class="chart-section">
            <h2>📈 Test run history</h2>
            <p class="chart-caption"><strong>How CI tracks this:</strong> each point is one GitHub Actions metrics deploy. The workflow pulls prior rows from <code>gh-pages</code> into <code>history-ci.jsonl</code> or <code>history-nightly.jsonl</code>, generates <code>latest-*.json</code>, then <strong>appends one compact JSON line</strong> for that run. The chart uses <strong>date + short commit</strong> on the x-axis so several runs on the same calendar day do not collapse into one dot. <strong>Left:</strong> total pytest wall time (s) for that run. <strong>Right:</strong> combined line coverage (%). Summary cards above show the <em>latest</em> snapshot only (test counts, flaky count, docstrings are not plotted here).</p>
            <div class="chart-container" id="wrap-trends-chart" style="display: none;">
                <canvas id="trendsChart"></canvas>
            </div>
            <div class="no-data chart-placeholder" id="placeholder-trends-chart" style="display: none;"></div>
        </div>

        <div class="chart-section">
            <h2>📊 Code quality history</h2>
            <p class="chart-caption"><strong>How CI tracks this:</strong> same <code>history-*.jsonl</code> points as <strong>Test run history</strong>—one row per metrics deploy. For each run, <code>generate_metrics.py</code> reads <strong>radon</strong> outputs in <code>reports/</code> (<code>complexity.json</code>, <code>maintainability.json</code>) and stores <strong>package-wide averages</strong> in the snapshot. <strong>Left:</strong> mean cyclomatic complexity. <strong>Right:</strong> mean maintainability index (MI). This line chart is <strong>not</strong> wily&rsquo;s per-commit file graph; for that, run <code>make complexity-track</code> locally—see <strong>CI → Code quality trends</strong>. <strong>Docstring %, dead-code count, and spelling</strong> from the same run appear in the summary cards only, not on this chart.</p>
            <div class="chart-container" id="wrap-quality-chart" style="display: none;">
                <canvas id="quality-chart"></canvas>
            </div>
            <div class="no-data chart-placeholder" id="placeholder-quality-chart" style="display: none;"></div>
        </div>

        <div id="pipeline-section" class="chart-section" style="display: none;">
            <h2>🚀 Sample pipeline run (history)</h2>
            <p class="chart-caption">Optional metrics from <code>collect_pipeline_metrics.py</code> when that step succeeds (often 1 episode in CI).</p>
            <div class="chart-container" id="wrap-pipeline-chart">
                <canvas id="pipeline-chart"></canvas>
            </div>
        </div>

        <div class="slowest-tests">
            <h2>🐌 Slowest Tests (Top __SLOWEST_TABLE_MAX__)</h2>
            <div id="slowest-tests-container"></div>
        </div>

        <div class="slowest-tests">
            <h2>⚠️ Flaky Tests</h2>
            <p>Tests that failed initially but passed on rerun</p>
            <div id="flaky-tests-container"></div>
        </div>
    </div>

    <script>
        let trendsChart, qualityChart, pipelineChart;
        let loadSeq = 0;

        /** X-axis label so multiple runs on the same day stay distinct (Chart.js category axis). */
        function historyChartLabel(h, i) {
            const day = (h.timestamp || '').substring(0, 10);
            const sha = (h.commit || '').trim().substring(0, 7);
            if (day && sha) return day + ' ' + sha;
            if (day) return day;
            return 'run ' + (i + 1);
        }

        /** Pull top-level JSON objects from text (legacy pretty-printed history blobs). */
        function extractTopLevelJsonObjects(text) {
            const out = [];
            const s = text;
            let i = 0;
            const n = s.length;
            while (i < n) {
                while (i < n && /\\s/.test(s[i])) i++;
                if (i >= n || s[i] !== '{') { i++; continue; }
                const start = i;
                let depth = 0;
                let inStr = false;
                let esc = false;
                for (; i < n; i++) {
                    const c = s[i];
                    if (esc) { esc = false; continue; }
                    if (inStr) {
                        if (c === '\\\\') esc = true;
                        else if (c === '"') inStr = false;
                        continue;
                    }
                    if (c === '"') { inStr = true; continue; }
                    if (c === '{') depth++;
                    else if (c === '}') {
                        depth--;
                        if (depth === 0) {
                            try {
                                const o = JSON.parse(s.slice(start, i + 1));
                                if (o && typeof o === 'object' && !Array.isArray(o)) out.push(o);
                            } catch (e) { /* skip */ }
                            i++;
                            break;
                        }
                    }
                }
            }
            return out;
        }

        function parseMetricsHistoryText(text) {
            const lines = text.split('\\n').map(l => l.trim()).filter(Boolean);
            const parsed = [];
            for (const line of lines) {
                try {
                    const o = JSON.parse(line);
                    if (o && typeof o === 'object' && !Array.isArray(o)) parsed.push(o);
                } catch (e) { /* skip bad line */ }
            }
            if (parsed.length > 0) return parsed;
            const t = text.trim();
            if (t.length > 0 && t[0] === '{') {
                const recovered = extractTopLevelJsonObjects(text);
                if (recovered.length > 0) return recovered;
                /* Some hosts store one pretty-printed object in *.jsonl (not JSONL). */
                try {
                    const one = JSON.parse(t);
                    if (one && typeof one === 'object' && !Array.isArray(one) && one.metrics) return [one];
                    if (Array.isArray(one)) return one.filter(x => x && typeof x === 'object');
                } catch (e) { /* ignore */ }
            }
            return [];
        }

        /** Merge latest run into history for charts when history is empty or stale (one blob file). */
        function chartHistoryFrom(latest, history) {
            const cap = 30;
            let h = history.slice(-cap);
            const last = h[h.length - 1];
            const sameCommit = last && latest.commit && last.commit === latest.commit;
            if (!sameCommit && latest && latest.metrics) {
                h = h.concat([latest]);
            }
            return h.slice(-cap);
        }

        function setDataBannerLoading() {
            const el = document.getElementById('data-mode-banner');
            if (!el) return;
            el.className = 'data-mode-banner data-mode-loading';
            el.textContent = 'Loading metrics…';
        }

        function setLoadModeBanner(bundle, source, latest, history) {
            const el = document.getElementById('data-mode-banner');
            if (!el) return;
            const hist = history || [];
            const chartHistory = latest && latest.metrics ? chartHistoryFrom(latest, hist) : [];
            const chartPts = chartHistory.length;
            const srcLabel = source === 'ci' ? 'CI' : 'Nightly';
            if (bundle) {
                const ciN = bundle.ci && bundle.ci.history ? bundle.ci.history.length : 0;
                const nyN = bundle.nightly && bundle.nightly.history ? bundle.nightly.history.length : 0;
                const gen = bundle.generated_at || '—';
                let html = (
                    '<strong>Unified data</strong> — loaded from <code>dashboard-data.json</code> (one request). '
                    + 'Bundle built <code>' + gen + '</code>. History rows in file: CI <strong>' + ciN + '</strong>, '
                    + 'Nightly <strong>' + nyN + '</strong>. You are viewing <strong>' + srcLabel + '</strong> '
                    + '(<strong>' + chartPts + '</strong> point' + (chartPts === 1 ? '' : 's') + ' on the charts below).'
                );
                if (source === 'nightly' && nyN <= 1) {
                    html += (
                        '<span class="banner-hint">Nightly trends stay flat until <code>history-nightly.jsonl</code> '
                        + 'has more rows—copy from GitHub Pages <code>metrics/</code> or wait for scheduled runs.</span>'
                    );
                }
                const m = latest && latest.metrics ? latest.metrics : null;
                const th = m && m.test_health ? m.test_health : null;
                const slow = m && Array.isArray(m.slowest_tests) ? m.slowest_tests : [];
                const totalT = th ? (th.total || 0) : 0;
                if (source === 'ci' && totalT > 100 && slow.length === 0) {
                    html += (
                        '<span class="banner-hint"><strong>Why slowest is empty:</strong> this <code>latest-ci.json</code> '
                        + 'was built without per-test timings in the bundle (older CI run or pre-fix snapshot). '
                        + 'The preview does not re-run <code>generate_metrics.py</code>; it only displays files under '
                        + '<code>artifacts/dashboard-preview/</code>. After fixes are on <code>main</code>, run '
                        + '<code>make fetch-ci-metrics</code> to download a newer metrics artifact, then '
                        + '<code>make build-metrics-dashboard-preview</code>.</span>'
                    );
                }
                if (source === 'nightly' && totalT > 100 && slow.length > 0 && slow.length < __SLOWEST_TABLE_MAX__) {
                    html += (
                        '<span class="banner-hint">Only <strong>' + slow.length + '</strong> slowest row(s) in this '
                        + 'snapshot—often sparse JSON before junit+merge. Newer nightly runs (or local rebuild) can '
                        + 'fill all <strong>' + __SLOWEST_TABLE_MAX__ + '</strong> slots in metrics JSON.</span>'
                    );
                }
                el.className = 'data-mode-banner data-mode-unified';
                el.innerHTML = html;
                return;
            }
            el.className = 'data-mode-banner data-mode-legacy';
            el.innerHTML = (
                '<strong>Legacy mode</strong> — separate fetches for <code>latest-' + source + '.json</code> and '
                + '<code>history-' + source + '.jsonl</code>. Viewing <strong>' + srcLabel + '</strong> (<strong>'
                + chartPts + '</strong> chart point' + (chartPts === 1 ? '' : 's') + '). '
                + '<span class="banner-hint">Run <code>make build-metrics-dashboard-preview</code> to produce '
                + '<code>dashboard-data.json</code> beside this page for unified single-fetch mode.</span>'
            );
        }

        let cachedBundle = undefined;

        async function tryLoadBundle() {
            if (cachedBundle !== undefined) return cachedBundle;
            try {
                const bust = '_=' + Date.now();
                const r = await fetch('dashboard-data.json?' + bust, { cache: 'no-store' });
                if (!r.ok) {
                    cachedBundle = null;
                    return null;
                }
                const j = await r.json();
                if (j && j.ci && j.nightly && j.ci.latest && j.nightly.latest
                        && Array.isArray(j.ci.history) && Array.isArray(j.nightly.history)) {
                    cachedBundle = j;
                    return j;
                }
                cachedBundle = null;
                return null;
            } catch (e) {
                console.warn('dashboard-data.json load failed, using legacy fetches', e);
                cachedBundle = null;
                return null;
            }
        }

        async function loadDataLegacy(source, seq) {
            const fetchOpts = { cache: 'no-store' };
            const bust = '_=' + Date.now();
            const [latestResp, historyResp] = await Promise.all([
                fetch(`latest-${source}.json?` + bust, fetchOpts),
                fetch(`history-${source}.jsonl?` + bust, fetchOpts)
            ]);

            if (seq !== loadSeq) return;

            if (!latestResp.ok) {
                showNoDataState(source, `Metrics file not found (${latestResp.status}). Data is generated after CI runs on main or scheduled nightly runs.`);
                return;
            }

            const rawLatest = await latestResp.text();
            if (seq !== loadSeq) return;
            let latest;
            try {
                latest = rawLatest.trim() ? JSON.parse(rawLatest) : null;
            } catch (parseErr) {
                console.error('Invalid JSON in latest-' + source + '.json:', parseErr);
                showNoDataState(source, 'Metrics file is empty or invalid. Data may not have been generated yet.');
                return;
            }

            if (!latest || typeof latest !== 'object') {
                showNoDataState(source, 'No metrics data available yet.');
                return;
            }

            let history = [];
            if (historyResp.ok) {
                const text = await historyResp.text();
                if (seq !== loadSeq) return;
                history = parseMetricsHistoryText(text);
            }

            if (seq !== loadSeq) return;
            setLoadModeBanner(null, source, latest, history);
            updateDashboard(latest, history);
        }

        async function loadData(source) {
            const seq = ++loadSeq;
            const loader = document.getElementById('loader');
            loader.style.display = 'block';
            setDataBannerLoading();

            try {
                const bundle = await tryLoadBundle();
                if (seq !== loadSeq) return;
                if (bundle) {
                    const slice = source === 'ci' ? bundle.ci : bundle.nightly;
                    const latest = slice.latest;
                    const history = slice.history || [];
                    if (!latest || typeof latest !== 'object') {
                        showNoDataState(source, 'Invalid bundle entry for ' + source + '.');
                        return;
                    }
                    setLoadModeBanner(bundle, source, latest, history);
                    updateDashboard(latest, history);
                    return;
                }
                await loadDataLegacy(source, seq);
            } catch (error) {
                console.error('Error loading data:', error);
                showNoDataState(source, error.message || 'Failed to load metrics.');
            } finally {
                loader.style.display = 'none';
            }
        }

        function showNoDataState(source, message) {
            const banner = document.getElementById('data-mode-banner');
            if (banner) {
                banner.className = 'data-mode-banner data-mode-error';
                banner.textContent = 'Metrics unavailable: ' + message;
            }
            document.getElementById('last-updated').textContent = 'No data';
            document.getElementById('commit-info').textContent = '';
            document.getElementById('workflow-link').textContent = '';
            document.getElementById('alerts-container').innerHTML =
                '<div class="alert alert-info"><strong>INFO</strong>: ' + message + '</div>';
            document.getElementById('metrics-grid').innerHTML =
                '<div class="no-data" style="grid-column: 1 / -1;">' + message + '</div>';
            document.getElementById('slowest-tests-container').innerHTML = '<div class="no-data">No data</div>';
            document.getElementById('flaky-tests-container').innerHTML = '<div class="no-data">No data</div>';
            if (trendsChart) { trendsChart.destroy(); trendsChart = null; }
            if (qualityChart) { qualityChart.destroy(); qualityChart = null; }
            if (pipelineChart) { pipelineChart.destroy(); pipelineChart = null; }
            document.getElementById('wrap-trends-chart').style.display = 'none';
            document.getElementById('placeholder-trends-chart').style.display = 'block';
            document.getElementById('placeholder-trends-chart').textContent = 'Charts unavailable until metrics load successfully.';
            document.getElementById('wrap-quality-chart').style.display = 'none';
            document.getElementById('placeholder-quality-chart').style.display = 'block';
            document.getElementById('placeholder-quality-chart').textContent = 'Charts unavailable until metrics load successfully.';
            document.getElementById('pipeline-section').style.display = 'none';
        }

        function formatTimestamp(ts) {
            if (!ts) return 'unknown';
            try {
                const dt = new Date(ts);
                return dt.toLocaleString();
            } catch (e) { return ts; }
        }

        function updateDashboard(latest, history) {
            const metrics = latest.metrics || {};
            const trends = latest.trends || {};
            const alerts = latest.alerts || [];

            // Update Header
            document.getElementById('last-updated').innerText = `Last updated: ${formatTimestamp(latest.timestamp)}`;
            document.getElementById('commit-info').innerHTML = `Commit: <code>${(latest.commit || 'unknown').substring(0, 8)}</code> | Branch: <code>${latest.branch || 'unknown'}</code>`;
            const wf = latest.workflow_run;
            const wfEl = document.getElementById('workflow-link');
            if (wf && typeof wf === 'string' && wf.indexOf('http') === 0) {
                wfEl.innerHTML = `Workflow run: <a href="${wf}" target="_blank" rel="noopener noreferrer">GitHub Actions</a>`;
            } else {
                wfEl.textContent = '';
            }

            // Update Alerts
            const alertsContainer = document.getElementById('alerts-container');
            if (alerts.length > 0) {
                alertsContainer.innerHTML = alerts.map(a => {
                    const severity = a.severity === 'error' ? 'alert-error' : (a.severity === 'warning' ? 'alert-warning' : 'alert-info');
                    return `<div class="alert ${severity}"><strong>${(a.metric || 'unknown').toUpperCase()}</strong>: ${a.message}</div>`;
                }).join('');
            } else {
                alertsContainer.innerHTML = '<div class="no-data">✅ No alerts - all metrics within normal range</div>';
            }

            // Update Metrics Grid
            const grid = document.getElementById('metrics-grid');
            const testHealth = metrics.test_health || {};
            const runtime = metrics.runtime || {};
            const coverage = metrics.coverage || {};
            const complexity = metrics.complexity || {};
            const pipeline = metrics.pipeline || {};

            let gridHtml = `
                <div class="metric-card">
                    <div class="metric-label">Total Tests</div>
                    <div class="metric-value">${testHealth.total || 0}</div>
                    ${trends.test_count_change ? `<div class="metric-trend">${trends.test_count_change}</div>` : ''}
                </div>
                <div class="metric-card">
                    <div class="metric-label">Passed</div>
                    <div class="metric-value" style="color: #27ae60;">${testHealth.passed || 0}</div>
                    <div class="metric-trend">Pass rate: ${(testHealth.pass_rate * 100 || 0).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Failed</div>
                    <div class="metric-value" style="color: #e74c3c;">${testHealth.failed || 0}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Skipped</div>
                    <div class="metric-value" style="color: #f39c12;">${testHealth.skipped || 0}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Runtime</div>
                    <div class="metric-value">${(runtime.total || 0).toFixed(1)}s</div>
                    ${getTrendHtml(trends.runtime_change)}
                </div>
                <div class="metric-card">
                    <div class="metric-label">Coverage (threshold: ${(coverage.threshold || 80).toFixed(0)}%)</div>
                    <div class="metric-value" style="color: ${coverage.meets_threshold ? '#27ae60' : '#e74c3c'};">${(coverage.overall || 0).toFixed(1)}%</div>
                    ${getTrendHtml(trends.coverage_change)}
                </div>
                <div class="metric-card">
                    <div class="metric-label">Complexity</div>
                    <div class="metric-value">${(complexity.cyclomatic_complexity || 0).toFixed(1)}</div>
                    ${getTrendHtml(complexity.complexity_trend)}
                </div>
                <div class="metric-card">
                    <div class="metric-label">Maintainability</div>
                    <div class="metric-value">${(complexity.maintainability_index || 0).toFixed(1)}</div>
                    ${getTrendHtml(complexity.maintainability_trend)}
                </div>
                <div class="metric-card">
                    <div class="metric-label">Docstrings</div>
                    <div class="metric-value">${(complexity.docstring_coverage || 0).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Dead Code</div>
                    <div class="metric-value" style="color: ${complexity.dead_code_count > 0 ? '#e74c3c' : '#27ae60'};">${complexity.dead_code_count || 0}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Spelling</div>
                    <div class="metric-value" style="color: ${complexity.spelling_errors_count > 0 ? '#e74c3c' : '#27ae60'};">${complexity.spelling_errors_count || 0}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Flaky Tests</div>
                    <div class="metric-value" style="color: ${testHealth.flaky > 0 ? '#e74c3c' : '#27ae60'};">${testHealth.flaky || 0}</div>
                </div>
            `;

            if (pipeline && pipeline.run_duration_seconds) {
                gridHtml += `
                    <div class="metric-card">
                        <div class="metric-label">Pipeline Duration</div>
                        <div class="metric-value">${pipeline.run_duration_seconds.toFixed(1)}s</div>
                        ${getTrendHtml(trends.pipeline_duration_change)}
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Episodes Scraped</div>
                        <div class="metric-value">${pipeline.episodes_scraped_total || 0}</div>
                        ${getTrendHtml(trends.pipeline_episodes_change)}
                    </div>
                `;
            }

            grid.innerHTML = gridHtml;

            // Update Tables
            updateTables(metrics);

            // Update Charts (include latest when it extends one-shot history files)
            updateCharts(latest, history);
        }

        function getTrendHtml(change) {
            if (change == null || change === '') return '';
            const s = String(change);
            const trendClass = s.startsWith('+') ? 'up' : (s.startsWith('-') ? 'down' : 'neutral');
            return `<div class="metric-trend trend-${trendClass}">${s}</div>`;
        }

        function formatTestDuration(seconds) {
            if (typeof seconds === 'number' && !Number.isNaN(seconds)) return seconds.toFixed(2) + 's';
            return '—';
        }

        function updateTables(metrics) {
            const slowest = metrics.slowest_tests || [];
            const flaky = (metrics.test_health || {}).flaky_tests || [];

            const slowestContainer = document.getElementById('slowest-tests-container');
            if (slowest.length > 0) {
                slowestContainer.innerHTML = `
                    <table>
                        <thead><tr><th>Test Name</th><th>Duration</th></tr></thead>
                        <tbody>${slowest.slice(0, __SLOWEST_TABLE_MAX__).map(t => `<tr><td><code>${(t.name || 'unknown')}</code></td><td>${formatTestDuration(t.duration)}</td></tr>`).join('')}</tbody>
                    </table>
                `;
            } else {
                slowestContainer.innerHTML = '<div class="no-data">No test duration data available</div>';
            }

            const flakyContainer = document.getElementById('flaky-tests-container');
            if (flaky.length > 0) {
                flakyContainer.innerHTML = `
                    <table>
                        <thead><tr><th>Test Name</th><th>Duration</th></tr></thead>
                        <tbody>${flaky.map(t => `<tr><td><code>${(t.name || 'unknown')}</code></td><td>${formatTestDuration(t.duration)}</td></tr>`).join('')}</tbody>
                    </table>
                `;
            } else {
                flakyContainer.innerHTML = '<div class="no-data">No tests passed after a pytest-rerunfailures retry in merged CI reports (unit, integration, and E2E use <code>--reruns 2</code> in GitHub Actions).</div>';
            }
        }

        function chartNum(h, pathFn) {
            const v = pathFn(h);
            return (typeof v === 'number' && !Number.isNaN(v)) ? v : null;
        }

        function updateCharts(latest, history) {
            const chartHistory = chartHistoryFrom(latest, history);
            const trendsWrap = document.getElementById('wrap-trends-chart');
            const trendsPh = document.getElementById('placeholder-trends-chart');
            const qualWrap = document.getElementById('wrap-quality-chart');
            const qualPh = document.getElementById('placeholder-quality-chart');
            const needHistoryMsg = 'No history entries. If you use dashboard-data.json, history arrays may be empty; otherwise check history-*.jsonl is JSONL (one compact JSON object per line). Summary cards still show the latest run.';
            const onePointMsg = 'Only one snapshot in history so far—trend lines will look flat until more nightly runs append rows.';

            if (chartHistory.length === 0) {
                if (trendsChart) { trendsChart.destroy(); trendsChart = null; }
                if (qualityChart) { qualityChart.destroy(); qualityChart = null; }
                if (pipelineChart) { pipelineChart.destroy(); pipelineChart = null; }
                trendsWrap.style.display = 'none';
                trendsPh.style.display = 'block';
                trendsPh.textContent = needHistoryMsg;
                qualWrap.style.display = 'none';
                qualPh.style.display = 'block';
                qualPh.textContent = needHistoryMsg;
                document.getElementById('pipeline-section').style.display = 'none';
                return;
            }

            trendsWrap.style.display = 'block';
            trendsPh.style.display = chartHistory.length === 1 ? 'block' : 'none';
            trendsPh.textContent = onePointMsg;
            qualWrap.style.display = 'block';
            qualPh.style.display = chartHistory.length === 1 ? 'block' : 'none';
            qualPh.textContent = onePointMsg;

            const labels = chartHistory.map((h, i) => historyChartLabel(h, i));

            // Test run history: two axes only (pytest duration + coverage %)
            const trendsData = {
                labels: labels,
                datasets: [
                    { label: 'Pytest duration (s)', data: chartHistory.map(h => chartNum(h, x => x.metrics && x.metrics.runtime && x.metrics.runtime.total)), borderColor: '#e74c3c', yAxisID: 'y', tension: 0.1 },
                    { label: 'Line coverage (%)', data: chartHistory.map(h => chartNum(h, x => x.metrics && x.metrics.coverage && x.metrics.coverage.overall)), borderColor: '#3498db', yAxisID: 'y1', tension: 0.1 }
                ]
            };

            if (trendsChart) trendsChart.destroy();
            trendsChart = new Chart(document.getElementById('trendsChart'), {
                type: 'line',
                data: trendsData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    scales: {
                        y: { type: 'linear', position: 'left', title: { display: true, text: 'Seconds' } },
                        y1: { type: 'linear', position: 'right', title: { display: true, text: 'Coverage %' }, grid: { drawOnChartArea: false } }
                    }
                }
            });

            // Code quality history: radon package averages (two axes)
            const qualityData = {
                labels: labels,
                datasets: [
                    { label: 'Cyclomatic complexity (avg)', data: chartHistory.map(h => chartNum(h, x => x.metrics && x.metrics.complexity && x.metrics.complexity.cyclomatic_complexity)), borderColor: '#e74c3c', yAxisID: 'y', tension: 0.1 },
                    { label: 'Maintainability index (avg)', data: chartHistory.map(h => chartNum(h, x => x.metrics && x.metrics.complexity && x.metrics.complexity.maintainability_index)), borderColor: '#2ecc71', yAxisID: 'y1', tension: 0.1 }
                ]
            };

            if (qualityChart) qualityChart.destroy();
            qualityChart = new Chart(document.getElementById('quality-chart'), {
                type: 'line',
                data: qualityData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    scales: {
                        y: { type: 'linear', position: 'left', title: { display: true, text: 'Complexity' } },
                        y1: { type: 'linear', position: 'right', title: { display: true, text: 'MI (0–100)' }, grid: { drawOnChartArea: false } }
                    }
                }
            });

            // Pipeline Trends
            const pipelineHistory = chartHistory.filter(h => h.metrics && h.metrics.pipeline && h.metrics.pipeline.run_duration_seconds);
            if (pipelineHistory.length > 0) {
                document.getElementById('pipeline-section').style.display = 'block';
                const pipelineLabels = pipelineHistory.map((h, i) => historyChartLabel(h, i));
                const pipelineData = {
                    labels: pipelineLabels,
                    datasets: [
                        { label: 'Duration (s)', data: pipelineHistory.map(h => chartNum(h, x => x.metrics && x.metrics.pipeline && x.metrics.pipeline.run_duration_seconds)), borderColor: '#9b59b6', yAxisID: 'y', tension: 0.1 },
                        { label: 'Episodes', data: pipelineHistory.map(h => chartNum(h, x => x.metrics && x.metrics.pipeline && x.metrics.pipeline.episodes_scraped_total)), borderColor: '#1abc9c', yAxisID: 'y1', tension: 0.1 }
                    ]
                };

                if (pipelineChart) pipelineChart.destroy();
                pipelineChart = new Chart(document.getElementById('pipeline-chart'), {
                    type: 'line',
                    data: pipelineData,
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { mode: 'index', intersect: false },
                        scales: {
                            y: { type: 'linear', position: 'left', title: { display: true, text: 'Seconds' } },
                            y1: { type: 'linear', position: 'right', title: { display: true, text: 'Count' }, grid: { drawOnChartArea: false } }
                        }
                    }
                });
            } else {
                document.getElementById('pipeline-section').style.display = 'none';
            }
        }

        document.getElementById('data-source').addEventListener('change', (e) => {
            loadData(e.target.value);
        });

        // Initial load
        loadData('ci');
    </script>
</body>
</html>
"""
    html = html.replace("__SLOWEST_TABLE_MAX__", str(SLOWEST_TESTS_TABLE_MAX))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"✅ Generated unified dashboard: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate HTML dashboard for test metrics")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("metrics/latest.json"),
        help="Path to latest metrics JSON (default: metrics/latest.json)",
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="Path to history.jsonl file (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/index.html"),
        help="Output path for HTML dashboard (default: metrics/index.html)",
    )
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Generate a unified dashboard with data source selector",
    )

    args = parser.parse_args()

    try:
        if args.unified:
            generate_unified_dashboard(output_path=args.output)
        else:
            generate_static_dashboard(
                metrics_path=args.metrics,
                history_path=args.history,
                output_path=args.output,
            )
    except Exception as e:
        print(f"❌ Error generating dashboard: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
