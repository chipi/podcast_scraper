# CI/CD & Quality Metrics

This section documents the Continuous Integration and Continuous Deployment (CI/CD) pipelines, as well as
the code quality metrics tracked for the Podcast Scraper project.

## Documentation

| Guide | Description |
| ------- | ------------- |
| [CI/CD Pipeline](CD.md) | Overview of GitHub Actions workflows, triggers, and execution flows |
| [Quality Metrics](QUALITY_METRICS.md) | Details on metrics collection, dashboards, and alert thresholds |

## Key Concepts

- **Tiered Testing**: Optimized execution flows for Pull Requests (fast) vs. Main Branch (comprehensive).
- **Parallel Execution**: Heavily parallelized jobs to minimize developer wait time.
- **Path-Based Optimization**: Workflows only trigger for relevant file changes.
- **Automated Dashboards**: Historical trend tracking for coverage, complexity, and performance.
