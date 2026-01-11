# ADR-022: Flaky Test Defense

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-025](../rfc/RFC-025-test-metrics-and-health-tracking.md)

## Context & Problem Statement

ML-based tests (Whisper, NER) and complex integration tests often suffer from intermittent "flakiness" due to timing, hardware variation, or model non-determinism. This causes CI to fail randomly, wasting developer time.

## Decision

We implement a **Flaky Test Defense** strategy:

1. **Automated Retries**: Critical tests are run with `pytest-rerunfailures` (max 3 retries) in CI.
2. **Visibility**: All failures (including successful retries) are recorded in JUnit XML and surfaced in the dashboard.
3. **Thresholds**: Any test failing >10% of the time is tagged as `@pytest.mark.flaky` and moved to a separate health-tracking tier.

## Rationale

- **CI Stability**: Prevents "false alarms" from blocking PRs.
- **Actionable Data**: Instead of just ignoring flakiness, we track it over time to identify which modules need stabilization work.

## Alternatives Considered

1. **Ignore Failures**: Rejected as it compromises the integrity of the test suite.
2. **Increase Timeouts**: Used as a secondary measure, but doesn't solve model non-determinism.

## Consequences

- **Positive**: Stable PR builds; clear visibility into test health.
- **Negative**: Can hide real (but intermittent) bugs if retries are used too aggressively.

## References

- [RFC-025: Test Metrics and Health Tracking](../rfc/RFC-025-test-metrics-and-health-tracking.md)
