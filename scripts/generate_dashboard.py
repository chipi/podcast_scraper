#!/usr/bin/env python3
"""
Generate HTML dashboard for test metrics visualization.

This script generates an interactive HTML dashboard that displays:
- Current metrics (latest run)
- Trend charts (last 30 runs)
- Deviation alerts
- Slowest tests
- Coverage trends

See RFC-026: Metrics Consumption and Dashboards
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def generate_dashboard(
    metrics_path: Path,
    history_path: Optional[Path],
    output_path: Path,
) -> None:
    """Generate HTML dashboard from metrics and history."""

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

    flaky_color = "#e74c3c" if test_health.get("flaky", 0) > 0 else "#27ae60"

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
        for test in slowest_tests[:10]:
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
                    {chr(10).join(test_rows)}
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
                    {chr(10).join(flaky_rows)}
                </tbody>
            </table>
        """
    else:
        flaky_tests_html = '<div class="no-data">✅ No flaky tests detected</div>'

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Metrics Dashboard - Podcast Scraper</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}

        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 8px;
        }}

        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}

        .metric-trend {{
            font-size: 0.9em;
            margin-top: 8px;
        }}

        .trend-up {{
            color: #27ae60;
        }}

        .trend-down {{
            color: #e74c3c;
        }}

        .trend-neutral {{
            color: #7f8c8d;
        }}

        .alerts-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        .alert {{
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
            border-left: 4px solid;
        }}

        .alert-error {{
            background: #fee;
            border-color: #e74c3c;
            color: #c0392b;
        }}

        .alert-warning {{
            background: #fffbf0;
            border-color: #f39c12;
            color: #d68910;
        }}

        .alert-info {{
            background: #ebf5fb;
            border-color: #3498db;
            color: #2874a6;
        }}

        .chart-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 20px;
        }}

        .slowest-tests {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}

        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}

        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}

        .no-data {{
            text-align: center;
            color: #7f8c8d;
            padding: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Test Metrics Dashboard</h1>
            <div class="timestamp">
                Last updated: {format_timestamp(latest.get("timestamp", ""))}
            </div>
            <div class="timestamp">
                Commit: <code>{latest.get("commit", "unknown")[:8]}</code> |
                Branch: <code>{latest.get("branch", "unknown")}</code>
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
                <div class="metric-label">Coverage</div>
                <div class="metric-value">{coverage.get("overall", 0):.1f}%</div>
                {coverage_trend}
            </div>

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
        </div>

        <div class="alerts-section">
            <h2>⚠️ Alerts</h2>
            {alerts_html}
        </div>

        <div class="chart-section">
            <h2>📈 Trends (Last {len(chart_history)} Runs)</h2>
            <div class="chart-container">
                <canvas id="trendsChart"></canvas>
            </div>
        </div>

        <div class="slowest-tests">
            <h2>🐌 Slowest Tests (Top 10)</h2>
            {slowest_tests_html}
        </div>

        <div class="slowest-tests">
            <h2>⚠️ Flaky Tests</h2>
            <p>Tests that failed initially but passed on rerun</p>
            {flaky_tests_html}
        </div>
    </div>

    <script>
        const ctx = document.getElementById('trendsChart').getContext('2d');
        const timestamps = {json.dumps(timestamps)};
        const runtimeData = {json.dumps(runtime_data)};
        const coverageData = {json.dumps(coverage_data)};
        const testCountData = {json.dumps(test_count_data)};
        const flakyData = {json.dumps(flaky_data)};

        new Chart(ctx, {{
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
    </script>
</body>
</html>
"""

    # Write HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"✅ Generated dashboard: {output_path}")
    print(f"   - Metrics from: {metrics_path}")
    print(f"   - History entries: {len(history)}")
    print(f"   - Alerts: {len(alerts)}")
    print(f"   - Slowest tests: {len(slowest_tests)}")
    print(f"   - Flaky tests: {len(flaky_tests)}")


def main():
    """Main entry point."""
    import argparse

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

    args = parser.parse_args()

    try:
        generate_dashboard(
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
