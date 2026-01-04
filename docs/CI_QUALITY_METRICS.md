# CI & Code Quality Metrics

This project tracks comprehensive CI/CD and code quality metrics via automated dashboards.

## Available Dashboards

### [📊 CI Metrics Dashboard](https://chipi.github.io/podcast_scraper/metrics/)

Real-time metrics from the latest CI run:

- **Test Results** — Passed, failed, skipped counts
- **Code Coverage** — Line and branch coverage percentages
- **Test Duration** — Total runtime and per-test timing
- **Code Quality** — Complexity, maintainability, docstring coverage
- **Alerts** — Automatic deviation detection from historical trends

### [🌙 Nightly Metrics](https://chipi.github.io/podcast_scraper/metrics/nightly/)

Comprehensive metrics from nightly full-suite runs:

- Full test suite results (all tiers)
- Production model performance
- Extended pipeline metrics

## Metrics Collection

Metrics are collected automatically:

| Source | Trigger | Data |
| -------- | --------- | ------ |
| `python-app.yml` | Every push to main | Test results, coverage, quality |
| `nightly.yml` | Daily schedule | Full suite, production models |

## Alert Thresholds

The dashboard automatically alerts when:

- Runtime increases >10% from historical median
- Coverage drops >1% from average
- Test count changes >5%
- Flaky tests increase significantly

See [RFC-025: Test Metrics and Health Tracking](rfc/RFC-025-test-metrics-and-health-tracking.md)
for implementation details.
