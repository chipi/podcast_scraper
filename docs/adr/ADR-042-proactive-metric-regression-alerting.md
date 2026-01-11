# ADR-042: Proactive Metric Regression Alerting

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-043](../rfc/RFC-043-automated-metrics-alerts.md)

## Context & Problem Statement

Performance metrics (runtime, coverage, WER) are collected in CI, but developers often had to manually check dashboards or job summaries to find them. This meant regressions were often missed until they became severe.

## Decision

We adopt **Proactive Metric Regression Alerting**.

- The CI pipeline automatically posts a **PR Comment** comparing the current branch's metrics against the `main` baseline.
- For regressions on the `main` branch (nightly runs), the system sends a **Webhook Notification** (Slack/Discord).
- Metrics tracked: Runtime changes (>10%), Coverage drops (>1%), and Flaky test counts.

## Rationale

- **High Visibility**: Regressions are brought directly into the developer's primary workspace (the PR thread).
- **Automation**: No manual comparison or "mental math" is required to see if a PR is slowing down the pipeline.
- **Transparency**: The entire team sees the performance impact of every change immediately.

## Alternatives Considered

1. **Manual Monitoring**: Rejected as it is inconsistent and reactive.
2. **Blocking CI Gates**: Rejected for now to allow developers to make informed tradeoffs (e.g., higher coverage at the cost of slightly higher runtime).

## Consequences

- **Positive**: Fast regression detection; zero manual overhead; improved performance culture.
- **Negative**: Requires managing GitHub API tokens and webhook secrets.

## References

- [RFC-043: Automated Metrics Alerts](../rfc/RFC-043-automated-metrics-alerts.md)
