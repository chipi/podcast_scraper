# ADR-023: Public Operational Metrics

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-026](../rfc/RFC-026-metrics-consumption-and-dashboards.md)

## Context & Problem Statement

Understanding the performance of the pipeline (WER, latency, cost, coverage) shouldn't require digging through GitHub Actions logs or running local analysis scripts.

## Decision

We adopt **Public Operational Metrics**:

1. CI jobs emit a standardized `metrics.json` artifact.
2. A dashboard generator converts these into a static HTML site hosted on **GitHub Pages**.
3. Metrics include: Test health, code coverage, and AI pipeline performance.

## Rationale

- **Transparency**: Provides an "at-a-glance" view of the project's health for all contributors.
- **Trend Analysis**: Static hosting allows us to track metrics over months without expensive database infrastructure.
- **Accessibility**: No special tools or environment required to view the "Scoreboard."

## Alternatives Considered

1. **Private Database (InfluxDB/Prometheus)**: Rejected as overkill and too complex for a single-dev/small-team project.

## Consequences

- **Positive**: High visibility; automated progress tracking for the "AI Quality Platform."
- **Negative**: Requires careful sanitization to ensure no private data (RSS URLs, filenames) leaks into public dashboards.

## References

- [RFC-026: Metrics Consumption and Dashboards](../rfc/RFC-026-metrics-consumption-and-dashboards.md)
