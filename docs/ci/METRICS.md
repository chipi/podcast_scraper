# CI & Code Quality Metrics

This project tracks comprehensive CI/CD and code quality metrics via automated dashboards.

## [📊 Unified Metrics Dashboard](https://chipi.github.io/podcast_scraper/metrics/)

A single interactive dashboard that displays metrics from both CI and Nightly runs with a data source selector.

### How It Works

The dashboard features a **data source dropdown** at the top that allows you to switch between:

- **CI Metrics (Latest Push)** — Real-time metrics from the latest CI run on main branch
- **Nightly Metrics (Scheduled Runs)** — Comprehensive metrics from scheduled nightly test runs

### Dashboard Features

Both data sources display the same comprehensive metrics:

- **Test Results** — Passed, failed, skipped counts, pass rate
- **Code Coverage** — Line and branch coverage percentages with threshold indicators
- **Test Duration** — Total runtime and tests per second
- **Code Quality** — Complexity, maintainability index, docstring coverage, dead code, spelling errors
- **Pipeline Metrics** — Run duration, episodes scraped, transcripts processed (when available)
- **LLM API Usage** — API call counts, token usage (input/output), and audio minutes for cost estimation
- **Alerts** — Automatic deviation detection from historical trends
- **Trend Charts** — Interactive charts showing last 30 runs for:
  - Runtime trends
  - Coverage trends
  - Test count changes
  - Flaky test counts
  - Code quality metrics
- **Slowest Tests** — Top 10 slowest tests with duration
- **Flaky Tests** — Tests that passed on rerun

### Data Source Differences

| Feature | CI Metrics | Nightly Metrics |
| --------- | ----------- | ---------------- |
| **Trigger** | Every push to main | Daily schedule (2 AM UTC) |
| **Test Suite** | Unit + Integration + E2E | Full suite + Nightly-only tests |
| **ML Models** | Test models | Production models |
| **Pipeline Metrics** | Basic (1 episode) | Extended (full pipeline) |
| **Update Frequency** | On every push | Once per day |

### Dashboard Metadata

The dashboard header displays:

- **Metrics collected** — Timestamp when metrics were collected
- **Dashboard built** — Timestamp when the HTML dashboard was generated
- **Workflow run** — Link to the GitHub Actions run that generated the metrics
- **Commit & Branch** — Git information for the metrics run

## Metrics Collection

Metrics are collected automatically by both workflows:

| Source | Trigger | Data Files | History File |
| -------- | --------- | ------ | ----------- |
| `python-app.yml` | Every push to main | `latest-ci.json` | `history-ci.jsonl` |
| `nightly.yml` | Daily schedule (2 AM UTC) | `latest-nightly.json` | `history-nightly.jsonl` |

Both workflows generate the same unified dashboard (`index.html`) that dynamically loads the
appropriate data files based on your selection.

### File Structure

```text
metrics/
├── index.html              # Unified dashboard (same for both workflows)
├── latest-ci.json          # CI metrics data
├── history-ci.jsonl        # CI history (one JSON object per line)
├── latest-nightly.json     # Nightly metrics data
└── history-nightly.jsonl   # Nightly history (one JSON object per line)
```python

## Alert Thresholds

The dashboard automatically alerts when:

- Runtime increases >10% from historical median
- Coverage drops >1% from average
- Test count changes >5%
- Flaky tests increase significantly

See [RFC-025: Test Metrics and Health Tracking](../rfc/RFC-025-test-metrics-and-health-tracking.md)
for implementation details.
