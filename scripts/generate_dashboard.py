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


def _generate_unified_dashboard(output_path: Path) -> None:
    """Generate unified dashboard that can switch between CI and Nightly data sources."""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
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
        }

        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
            margin: 5px 0;
        }

        .data-source-selector {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .data-source-selector label {
            font-weight: 600;
            color: #2c3e50;
        }

        .data-source-selector select {
            padding: 8px 12px;
            border: 2px solid #bdc3c7;
            border-radius: 4px;
            font-size: 1em;
            background: white;
            color: #2c3e50;
            cursor: pointer;
            min-width: 200px;
        }

        .data-source-selector select:hover {
            border-color: #3498db;
        }

        .data-source-selector select:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }

        .error-message {
            background: #fee;
            color: #c0392b;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #e74c3c;
            margin: 20px 0;
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

        .alerts-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
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
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Test Metrics Dashboard</h1>
            <div class="timestamp" id="metrics-timestamp">Select a data source to view metrics</div>
            <div class="timestamp" id="dashboard-timestamp"></div>
            <div class="timestamp" id="workflow-info"></div>
        </header>

        <div class="data-source-selector">
            <label for="dataSource">Data Source:</label>
            <select id="dataSource" onchange="loadDataSource()">
                <option value="">-- Select Data Source --</option>
                <option value="ci">CI Metrics (Latest Push)</option>
                <option value="nightly">Nightly Metrics (Scheduled Runs)</option>
            </select>
            <span id="dataSourceStatus"></span>
        </div>

        <div id="dashboardContent">
            <div class="loading">Select a data source from the dropdown above to view metrics</div>
        </div>
    </div>

    <script>
        const dashboardBuildTime = new Date().toISOString();
        document.getElementById('dashboard-timestamp').textContent = (
            'Dashboard built: ' + new Date(dashboardBuildTime).toLocaleString()
        );

        let currentCharts = [];

        async function loadDataSource() {
            const source = document.getElementById('dataSource').value;
            const statusEl = document.getElementById('dataSourceStatus');
            const contentEl = document.getElementById('dashboardContent');

            if (!source) {
                contentEl.innerHTML = (
                    '<div class="loading">Select a data source from the dropdown '
                    'above to view metrics</div>'
                );
                statusEl.textContent = '';
                return;
            }

            statusEl.textContent = 'Loading...';
            contentEl.innerHTML = '<div class="loading">Loading metrics data...</div>';

            // Destroy existing charts
            currentCharts.forEach(chart => chart.destroy());
            currentCharts = [];

            try {
                // Determine file paths based on source
                const metricsFile = source === 'ci' ? 'latest-ci.json' : 'latest-nightly.json';
                const historyFile = source === 'ci' ? 'history-ci.jsonl' : 'history-nightly.jsonl';

                // Load latest metrics
                const metricsResponse = await fetch(metricsFile);
                if (!metricsResponse.ok) {
                    throw new Error(`Failed to load ${metricsFile}: ${metricsResponse.statusText}`);
                }
                const latest = await metricsResponse.json();

                // Load history
                let history = [];
                try {
                    const historyResponse = await fetch(historyFile);
                    if (historyResponse.ok) {
                        const historyText = await historyResponse.text();
                        history = historyText.trim().split(/\\r?\\n/)
                            .filter(line => line.trim())
                            .map(line => {
                                try {
                                    return JSON.parse(line);
                                } catch (e) {
                                    console.warn('Failed to parse history line:', e);
                                    return null;
                                }
                            })
                            .filter(item => item !== null);
                        // Include latest if not already in history
                        if (history.length === 0 || (
                            history[history.length - 1].timestamp !== latest.timestamp
                        )) {
                            history.push(latest);
                        }
                    }
                } catch (e) {
                    console.warn('Could not load history:', e);
                    history = [latest];
                }

                // Update header
                const metricsTime = new Date(latest.timestamp).toLocaleString();
                document.getElementById('metrics-timestamp').innerHTML =
                    '<strong>Metrics collected:</strong> ' + metricsTime;

                const workflowLink = latest.workflow_run ?
                    '<a href="' + latest.workflow_run + '" target="_blank">View Run</a>' : 'N/A';
                document.getElementById('workflow-info').innerHTML = (
                    '<strong>Workflow run:</strong> ' + workflowLink + ' | ' +
                    '<strong>Commit:</strong> <code>' +
                    (latest.commit || 'unknown').substring(0, 8) + '</code> | ' +
                    '<strong>Branch:</strong> <code>' +
                    (latest.branch || 'unknown') + '</code>'
                );

                // Render dashboard
                renderDashboard(latest, history);
                statusEl.textContent = '✅ Loaded';
            } catch (error) {
                console.error('Error loading data:', error);
                contentEl.innerHTML = '<div class="error-message"><strong>Error:</strong> ' +
                    error.message + '<br>Make sure the metrics files are available.</div>';
                statusEl.textContent = '❌ Error';
            }
        }

        function renderDashboard(latest, history) {
            const metrics = latest.metrics || {};
            const runtime = metrics.runtime || {};
            const testHealth = metrics.test_health || {};
            const coverage = metrics.coverage || {};
            const complexity = metrics.complexity || {};
            const pipelineMetrics = metrics.pipeline || {};
            const slowestTests = metrics.slowest_tests || [];
            const flakyTests = testHealth.flaky_tests || [];
            const trends = latest.trends || {};
            const alerts = latest.alerts || [];

            // Prepare chart data
            const chartHistory = history.slice(-30);
            const timestamps = chartHistory.map(
                h => h.timestamp ? h.timestamp.substring(0, 10) : ''
            );
            const runtimeData = chartHistory.map(
                h => (h.metrics || {}).runtime?.total || 0
            );
            const coverageData = chartHistory.map(
                h => (h.metrics || {}).coverage?.overall || 0
            );
            const testCountData = chartHistory.map(
                h => (h.metrics || {}).test_health?.total || 0
            );
            const flakyData = chartHistory.map(
                h => (h.metrics || {}).test_health?.flaky || 0
            );
            const pipelineRunDurationData = chartHistory.map(
                h => (h.metrics || {}).pipeline?.run_duration_seconds || 0
            );
            const pipelineEpisodesScrapedData = chartHistory.map(
                h => (h.metrics || {}).pipeline?.episodes_scraped_total || 0
            );
            const buildDurationData = chartHistory.map(
                h => (h.metrics || {}).build?.total_duration_seconds || 0
            );
            const complexityData = chartHistory.map(
                h => (h.metrics || {}).complexity?.cyclomatic_complexity || 0
            );
            const maintainabilityData = chartHistory.map(
                h => (h.metrics || {}).complexity?.maintainability_index || 0
            );
            const docstringsData = chartHistory.map(
                h => (h.metrics || {}).complexity?.docstring_coverage || 0
            );
            const deadCodeData = chartHistory.map(
                h => (h.metrics || {}).complexity?.dead_code_count || 0
            );
            const spellingErrorsData = chartHistory.map(
                h => (h.metrics || {}).complexity?.spelling_errors_count || 0
            );

            // Build HTML
            let html = '';

            // Alerts
            if (alerts.length > 0) {
                html += '<div class="alerts-section"><h2>⚠️ Alerts</h2>';
                alerts.forEach(alert => {
                    const severityClass = alert.severity === 'error' ? (
                        'alert-error'
                    ) : alert.severity === 'warning' ? (
                        'alert-warning'
                    ) : 'alert-info';
                    html += (
                        `<div class="alert ${severityClass}">`
                        `<strong>${alert.metric.toUpperCase()}</strong>: `
                        `${alert.message}</div>`
                    );
                });
                html += '</div>';
            } else {
                html += (
                    '<div class="alerts-section"><h2>⚠️ Alerts</h2>'
                    '<div class="no-data">✅ No alerts - all metrics within '
                    'normal range</div></div>'
                );
            }

            // Build Metrics Section
            const buildMetrics = metrics.build || {};
            const buildDuration = buildMetrics.total_duration_seconds || runtime.total || 0;
            const workflowName = buildMetrics.workflow_name || 'unknown';

            html += '<div class="chart-section"><h2>🔧 Build Metrics</h2>';
            html += '<div class="metrics-grid">';
            html += (
                '<div class="metric-card">'
                '<div class="metric-label">Total Build Duration</div>'
                '<div class="metric-value">'
                `${(buildDuration / 60).toFixed(1)}m</div>`
                '<div class="metric-trend" style="font-size: 0.8em; '
                'color: #7f8c8d;">'
                `${buildDuration.toFixed(1)}s</div></div>`
            );
            html += (
                '<div class="metric-card">'
                '<div class="metric-label">Workflow</div>'
                '<div class="metric-value" style="font-size: 1.2em;">'
                `${workflowName}</div></div>`
            );
            html += '</div></div>';

            // Test Metrics Section
            html += '<div class="chart-section"><h2>🧪 Test Metrics</h2>';
            html += '<div class="metrics-grid">';
            html += (
                '<div class="metric-card">'
                '<div class="metric-label">Total Tests</div>'
                '<div class="metric-value">'
                `${testHealth.total || 0}</div></div>`
            );
            html += (
                '<div class="metric-card">'
                '<div class="metric-label">Passed</div>'
                '<div class="metric-value" style="color: #27ae60;">'
                `${testHealth.passed || 0}</div></div>`
            );
            html += (
                '<div class="metric-card">'
                '<div class="metric-label">Failed</div>'
                '<div class="metric-value" style="color: #e74c3c;">'
                `${testHealth.failed || 0}</div></div>`
            );
            html += (
                '<div class="metric-card">'
                '<div class="metric-label">Test Runtime</div>'
                '<div class="metric-value">'
                `${(runtime.total || 0).toFixed(1)}s</div></div>`
            );
            html += (
                '<div class="metric-card">'
                '<div class="metric-label">Coverage</div>'
                '<div class="metric-value" style="color: '
                `${coverage.meets_threshold ? '#27ae60' : '#e74c3c'};`
                '">'
                `${(coverage.overall || 0).toFixed(1)}%</div></div>`
            );
            html += (
                '<div class="metric-card">'
                '<div class="metric-label">Flaky Tests</div>'
                '<div class="metric-value" style="color: '
                `${testHealth.flaky > 0 ? '#e74c3c' : '#27ae60'};`
                '">'
                `${testHealth.flaky || 0}</div></div>`
            );
            html += '</div></div>';

            // Code Quality Metrics Section
            if (complexity && Object.keys(complexity).length > 0) {
                const cyclomaticComplexity = complexity.cyclomatic_complexity || 0;
                const maintainabilityIndex = complexity.maintainability_index || 0;
                const docstringCoverage = complexity.docstring_coverage || 0;
                const deadCodeCount = complexity.dead_code_count || 0;
                const spellingErrors = complexity.spelling_errors_count || 0;

                html += '<div class="chart-section"><h2>📊 Code Quality Metrics</h2>';
                html += '<div class="metrics-grid">';
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Cyclomatic Complexity</div>'
                    '<div class="metric-value">'
                    `${cyclomaticComplexity.toFixed(1)}</div></div>`
                );
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Maintainability Index</div>'
                    '<div class="metric-value">'
                    `${maintainabilityIndex.toFixed(1)}</div></div>`
                );
                const docstringColor = docstringCoverage >= 80 ? (
                    '#27ae60'
                ) : docstringCoverage >= 60 ? (
                    '#f39c12'
                ) : '#e74c3c';
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Docstring Coverage</div>'
                    '<div class="metric-value" style="color: '
                    `${docstringColor};">`
                    `${docstringCoverage.toFixed(1)}%</div></div>`
                );
                const deadCodeColor = deadCodeCount > 0 ? '#e74c3c' : '#27ae60';
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Dead Code Items</div>'
                    '<div class="metric-value" style="color: '
                    `${deadCodeColor};">`
                    `${deadCodeCount}</div></div>`
                );
                const spellingColor = spellingErrors > 0 ? '#e74c3c' : '#27ae60';
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Spelling Errors</div>'
                    '<div class="metric-value" style="color: '
                    `${spellingColor};">`
                    `${spellingErrors}</div></div>`
                );
                html += '</div></div>';
            }

            // Pipeline Performance Metrics Section (if available)
            if (pipelineMetrics && Object.keys(pipelineMetrics).length > 0) {
                const runDuration = pipelineMetrics.run_duration_seconds || 0;
                const episodesScraped = pipelineMetrics.episodes_scraped_total || 0;
                const episodesSkipped = pipelineMetrics.episodes_skipped_total || 0;
                const transcriptsDownloaded = pipelineMetrics.transcripts_downloaded || 0;
                const transcriptsTranscribed = pipelineMetrics.transcripts_transcribed || 0;
                const episodesSummarized = pipelineMetrics.episodes_summarized || 0;
                const metadataFiles = pipelineMetrics.metadata_files_generated || 0;

                html += '<div class="chart-section"><h2>🚀 Pipeline Performance Metrics</h2>';
                html += '<div class="metrics-grid">';
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Pipeline Run Duration</div>'
                    '<div class="metric-value">'
                    `${runDuration.toFixed(1)}s</div></div>`
                );
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Episodes Scraped</div>'
                    '<div class="metric-value">'
                    `${episodesScraped}</div></div>`
                );
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Episodes Skipped</div>'
                    '<div class="metric-value">'
                    `${episodesSkipped}</div></div>`
                );
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Transcripts Downloaded</div>'
                    '<div class="metric-value">'
                    `${transcriptsDownloaded}</div></div>`
                );
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Transcripts Transcribed</div>'
                    '<div class="metric-value">'
                    `${transcriptsTranscribed}</div></div>`
                );
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Episodes Summarized</div>'
                    '<div class="metric-value">'
                    `${episodesSummarized}</div></div>`
                );
                html += (
                    '<div class="metric-card">'
                    '<div class="metric-label">Metadata Files</div>'
                    '<div class="metric-value">'
                    `${metadataFiles}</div></div>`
                );
                html += '</div></div>';
            }

            // Charts - in same order as metrics sections
            // 1. Build Metrics → Build Duration Trends
            html += (
                '<div class="chart-section">'
                '<h2>⏱️ Build Duration Trends (Last ' +
                chartHistory.length + ' Runs)</h2>'
            );
            html += (
                '<div class="chart-container">'
                '<canvas id="buildChart"></canvas></div></div>'
            );

            // 2. Test Metrics → Test Metrics Trends
            html += (
                '<div class="chart-section">'
                '<h2>📈 Test Metrics Trends (Last ' +
                chartHistory.length + ' Runs)</h2>'
            );
            html += (
                '<div class="chart-container">'
                '<canvas id="trendsChart"></canvas></div></div>'
            );

            // 3. Code Quality Metrics → Code Quality Trends
            if (complexity && Object.keys(complexity).length > 0) {
                html += (
                    '<div class="chart-section">'
                    '<h2>📊 Code Quality Trends (Last ' +
                    chartHistory.length + ' Runs)</h2>'
                );
                html += (
                    '<div class="chart-container">'
                    '<canvas id="qualityChart"></canvas></div></div>'
                );
            }

            // 4. Pipeline Performance Metrics → Pipeline Performance Trends
            if (pipelineMetrics && Object.keys(pipelineMetrics).length > 0) {
                html += (
                    '<div class="chart-section">'
                    '<h2>🚀 Pipeline Performance Trends (Last ' +
                    chartHistory.length + ' Runs)</h2>'
                );
                html += (
                    '<div class="chart-container">'
                    '<canvas id="pipelineChart"></canvas></div></div>'
                );
            }

            // Slowest tests
            html += '<div class="slowest-tests"><h2>🐌 Slowest Tests (Top 10)</h2>';
            if (slowestTests.length > 0) {
                html += '<table><thead><tr><th>Test Name</th><th>Duration</th></tr></thead><tbody>';
                slowestTests.slice(0, 10).forEach(test => {
                    html += (
                        '<tr><td><code>'
                        `${test.name}</code></td><td>`
                        `${test.duration.toFixed(2)}s</td></tr>`
                    );
                });
                html += '</tbody></table></div>';
            } else {
                html += '<div class="no-data">No test duration data available</div></div>';
            }

            document.getElementById('dashboardContent').innerHTML = html;

            // Render charts - in same order as metrics sections
            setTimeout(() => {
                // 1. Build Duration Chart
                const buildCtx = document.getElementById('buildChart');
                if (buildCtx) {
                    const buildChart = new Chart(buildCtx.getContext('2d'), {
                        type: 'line',
                        data: {
                            labels: timestamps,
                            datasets: [{
                                label: 'Build Duration (minutes)',
                                data: buildDurationData.map(d => d / 60), // Convert to minutes
                                borderColor: 'rgb(52, 152, 219)',
                                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                                tension: 0.1,
                                fill: true
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            interaction: { mode: 'index', intersect: false },
                            plugins: {
                                legend: { display: true, position: 'top' },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            const minutes = context.parsed.y;
                                            const seconds = buildDurationData[context.dataIndex];
                                            return (
                                                `Build Duration: ${minutes.toFixed(1)}m `
                                                `(${seconds.toFixed(1)}s)`
                                            );
                                        }
                                    }
                                }
                            },
                            scales: {
                                y: {
                                    type: 'linear',
                                    display: true,
                                    position: 'left',
                                    title: { display: true, text: 'Duration (minutes)' },
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                    currentCharts.push(buildChart);
                }

                // 2. Test Metrics Chart
                const ctx = document.getElementById('trendsChart');
                if (ctx) {
                    const chart = new Chart(ctx.getContext('2d'), {
                        type: 'line',
                        data: {
                            labels: timestamps,
                            datasets: [
                                {
                                    label: 'Runtime (s)',
                                    data: runtimeData,
                                    borderColor: 'rgb(231, 76, 60)',
                                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                                    yAxisID: 'y',
                                    tension: 0.1
                                },
                                {
                                    label: 'Coverage (%)',
                                    data: coverageData,
                                    borderColor: 'rgb(52, 152, 219)',
                                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                                    yAxisID: 'y1',
                                    tension: 0.1
                                },
                                {
                                    label: 'Test Count',
                                    data: testCountData,
                                    borderColor: 'rgb(46, 204, 113)',
                                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                                    yAxisID: 'y2',
                                    tension: 0.1
                                },
                                {
                                    label: 'Flaky Tests',
                                    data: flakyData,
                                    borderColor: 'rgb(243, 156, 18)',
                                    backgroundColor: 'rgba(243, 156, 18, 0.1)',
                                    yAxisID: 'y2',
                                    tension: 0.1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            interaction: { mode: 'index', intersect: false },
                            plugins: { legend: { display: true, position: 'top' } },
                            scales: {
                                y: {
                                    type: 'linear',
                                    display: true,
                                    position: 'left',
                                    title: { display: true, text: 'Runtime (seconds)' }
                                },
                                y1: {
                                    type: 'linear',
                                    display: true,
                                    position: 'right',
                                    title: { display: true, text: 'Coverage (%)' },
                                    grid: { drawOnChartArea: false }
                                },
                                y2: { type: 'linear', display: false }
                            }
                        }
                    });
                    currentCharts.push(chart);
                }

                // 3. Code Quality Chart
                if (complexity && Object.keys(complexity).length > 0) {
                    const qualityCtx = document.getElementById('qualityChart');
                    if (qualityCtx) {
                        const qualityChart = new Chart(qualityCtx.getContext('2d'), {
                            type: 'line',
                            data: {
                                labels: timestamps,
                                datasets: [
                                    {
                                        label: 'Cyclomatic Complexity',
                                        data: complexityData,
                                        borderColor: 'rgb(155, 89, 182)',
                                        backgroundColor: 'rgba(155, 89, 182, 0.1)',
                                        yAxisID: 'y',
                                        tension: 0.1
                                    },
                                    {
                                        label: 'Maintainability Index',
                                        data: maintainabilityData,
                                        borderColor: 'rgb(26, 188, 156)',
                                        backgroundColor: 'rgba(26, 188, 156, 0.1)',
                                        yAxisID: 'y1',
                                        tension: 0.1
                                    },
                                    {
                                        label: 'Docstring Coverage (%)',
                                        data: docstringsData,
                                        borderColor: 'rgb(52, 152, 219)',
                                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                                        yAxisID: 'y1',
                                        tension: 0.1
                                    },
                                    {
                                        label: 'Dead Code Items',
                                        data: deadCodeData,
                                        borderColor: 'rgb(231, 76, 60)',
                                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                                        yAxisID: 'y2',
                                        tension: 0.1
                                    },
                                    {
                                        label: 'Spelling Errors',
                                        data: spellingErrorsData,
                                        borderColor: 'rgb(243, 156, 18)',
                                        backgroundColor: 'rgba(243, 156, 18, 0.1)',
                                        yAxisID: 'y2',
                                        tension: 0.1
                                    }
                                ]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                interaction: { mode: 'index', intersect: false },
                                plugins: { legend: { display: true, position: 'top' } },
                                scales: {
                                    y: {
                                        type: 'linear',
                                        display: true,
                                        position: 'left',
                                        title: { display: true, text: 'Cyclomatic Complexity' }
                                    },
                                    y1: {
                                        type: 'linear',
                                        display: true,
                                        position: 'right',
                                        title: {
                                            display: true,
                                            text: 'Maintainability / Docstring Coverage'
                                        },
                                        grid: { drawOnChartArea: false }
                                    },
                                    y2: { type: 'linear', display: false }
                                }
                            }
                        });
                        currentCharts.push(qualityChart);
                    }
                }

                // 4. Pipeline Performance Chart
                if (pipelineMetrics && Object.keys(pipelineMetrics).length > 0) {
                    const pipelineCtx = document.getElementById('pipelineChart');
                    if (pipelineCtx) {
                        const pipelineChart = new Chart(pipelineCtx.getContext('2d'), {
                            type: 'line',
                            data: {
                                labels: timestamps,
                                datasets: [
                                    {
                                        label: 'Run Duration (s)',
                                        data: pipelineRunDurationData,
                                        borderColor: 'rgb(155, 89, 182)',
                                        backgroundColor: 'rgba(155, 89, 182, 0.1)',
                                        yAxisID: 'y',
                                        tension: 0.1
                                    },
                                    {
                                        label: 'Episodes Scraped',
                                        data: pipelineEpisodesScrapedData,
                                        borderColor: 'rgb(26, 188, 156)',
                                        backgroundColor: 'rgba(26, 188, 156, 0.1)',
                                        yAxisID: 'y1',
                                        tension: 0.1
                                    }
                                ]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                interaction: { mode: 'index', intersect: false },
                                plugins: { legend: { display: true, position: 'top' } },
                                scales: {
                                    y: {
                                        type: 'linear',
                                        display: true,
                                        position: 'left',
                                        title: { display: true, text: 'Run Duration (seconds)' }
                                    },
                                    y1: {
                                        type: 'linear',
                                        display: true,
                                        position: 'right',
                                        title: { display: true, text: 'Episodes Scraped' },
                                        grid: { drawOnChartArea: false }
                                    }
                                }
                            }
                        });
                        currentCharts.push(pipelineChart);
                    }
                }
            }, 100);
        }

        // Auto-select CI if available
        window.addEventListener('load', () => {
            fetch('latest-ci.json')
                .then(r => r.ok ? document.getElementById('dataSource').value = 'ci' : null)
                .then(() => loadDataSource())
                .catch(() => {
                    // Try nightly if CI not available
                    fetch('latest-nightly.json')
                        .then(r => {
                            if (r.ok) {
                                document.getElementById('dataSource').value = 'nightly';
                            }
                            return null;
                        })
                        .then(() => loadDataSource())
                        .catch(() => {});
                });
        });
    </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"✅ Generated unified dashboard: {output_path}")
    print("   - Supports CI and Nightly data sources")
    print("   - Auto-detects available data sources")


def generate_dashboard(
    metrics_path: Path,
    history_path: Optional[Path],
    output_path: Path,
    support_multiple_sources: bool = False,
) -> None:
    """Generate HTML dashboard from metrics and history.

    Args:
        metrics_path: Path to latest metrics JSON (or default for multi-source mode)
        history_path: Path to history.jsonl file (optional)
        output_path: Output path for HTML dashboard
        support_multiple_sources: If True, creates a unified dashboard
            that can switch between CI and Nightly data
    """

    # For multi-source mode, we'll create a dashboard that loads data dynamically
    if support_multiple_sources:
        _generate_unified_dashboard(output_path)
        return

    # Load data (single source mode - original behavior)
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
    else:
        pipeline_metrics_html = ""

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
        pipeline_chart_code = """
        const pipelineCtx = document.getElementById('pipeline-chart');
        if (pipelineCtx) {
            new Chart(pipelineCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: timestamps,
                    datasets: [
                        {
                            label: 'Run Duration (s)',
                            data: pipelineRunDurationData,
                            borderColor: 'rgb(155, 89, 182)',
                            backgroundColor: 'rgba(155, 89, 182, 0.1)',
                            yAxisID: 'y',
                            tension: 0.1
                        },
                        {
                            label: 'Episodes Scraped',
                            data: pipelineEpisodesScrapedData,
                            borderColor: 'rgb(26, 188, 156)',
                            backgroundColor: 'rgba(26, 188, 156, 0.1)',
                            yAxisID: 'y1',
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                        },
                        tooltip: {
                            enabled: true,
                        }
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Run Duration (seconds)'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Episodes Scraped'
                            },
                            grid: {
                                drawOnChartArea: false,
                            }
                        }
                    }
                }
            });
        }
        """
    else:
        pipeline_performance_section = ""
        pipeline_chart_code = ""

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

    # Get dashboard build timestamp (when HTML is generated)
    dashboard_build_timestamp = datetime.utcnow().isoformat() + "Z"

    # Get workflow run URL for linking
    workflow_run_url = latest.get("workflow_run", "")
    workflow_run_link = (
        f'<a href="{workflow_run_url}" target="_blank">View Run</a>' if workflow_run_url else "N/A"
    )

    # Generate HTML
    html = f"""<!DOCTYPE html>  # noqa: E501
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

        .data-source-selector {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .data-source-selector label {{
            font-weight: 600;
            color: #2c3e50;
        }}

        .data-source-selector select {{
            padding: 8px 12px;
            border: 2px solid #bdc3c7;
            border-radius: 4px;
            font-size: 1em;
            background: white;
            color: #2c3e50;
            cursor: pointer;
        }}

        .data-source-selector select:hover {{
            border-color: #3498db;
        }}

        .data-source-selector select:focus {{
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }}

        .loading {{
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }}

        .error-message {{
            background: #fee;
            color: #c0392b;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #e74c3c;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Test Metrics Dashboard</h1>
            <div class="timestamp">
                <strong>Metrics collected:</strong> {format_timestamp(latest.get("timestamp", ""))}
            </div>
            <div class="timestamp">
                <strong>Dashboard built:</strong> {format_timestamp(dashboard_build_timestamp)}
            </div>
            <div class="timestamp">
                <strong>Workflow run:</strong> {workflow_run_link} |
                <strong>Commit:</strong> <code>{latest.get("commit", "unknown")[:8]}</code> |
                <strong>Branch:</strong> <code>{latest.get("branch", "unknown")}</code>
            </div>
        </header>

        <div class="alerts-section">
            <h2>⚠️ Alerts</h2>
            {alerts_html}
        </div>

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
                <div class="metric-label">Coverage (threshold: {coverage.get("threshold", 75):.0f}%)</div>
                <div class="metric-value" style="color: {'#27ae60' if coverage.get('meets_threshold', True) else '#e74c3c'};">
                    {coverage.get("overall", 0):.1f}%
                </div>
                {coverage_trend}
            </div>

            <div class="metric-card">
                <div class="metric-label">Cyclomatic Complexity</div>
                <div class="metric-value">{complexity.get("cyclomatic_complexity", 0):.1f}</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Maintainability Index</div>
                <div class="metric-value">{complexity.get("maintainability_index", 0):.1f}</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Docstring Coverage</div>
                <div class="metric-value">{complexity.get("docstring_coverage", 0):.1f}%</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Dead Code Items</div>
                <div class="metric-value" style="color: {'#e74c3c' if complexity.get('dead_code_count', 0) > 0 else '#27ae60'};">
                    {complexity.get("dead_code_count", 0)}
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Spelling Errors</div>
                <div class="metric-value" style="color: {'#e74c3c' if complexity.get('spelling_errors_count', 0) > 0 else '#27ae60'};">
                    {complexity.get("spelling_errors_count", 0)}
                </div>
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
        const complexityData = {json.dumps(complexity_data)};
        const maintainabilityData = {json.dumps(maintainability_data)};
        const docstringsData = {json.dumps(docstrings_data)};
        const deadCodeData = {json.dumps(dead_code_data)};
        const spellingErrorsData = {json.dumps(spelling_errors_data)};
        const flakyData = {json.dumps(flaky_data)};
        const pipelineRunDurationData = {json.dumps(pipeline_run_duration_data)};
        const pipelineEpisodesScrapedData = {json.dumps(pipeline_episodes_scraped_data)};

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

        // Pipeline Performance Chart (only if pipeline metrics available)
        {pipeline_chart_code}
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
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Generate unified dashboard that supports multiple data sources (CI and Nightly)",
    )

    args = parser.parse_args()

    try:
        generate_dashboard(
            metrics_path=args.metrics,
            history_path=args.history,
            output_path=args.output,
            support_multiple_sources=args.unified,
        )
    except Exception as e:
        print(f"❌ Error generating dashboard: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
