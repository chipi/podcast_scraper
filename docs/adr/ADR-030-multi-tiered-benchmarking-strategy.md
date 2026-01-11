# ADR-030: Multi-Tiered Benchmarking Strategy

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md)

## Context & Problem Statement

Running a comprehensive benchmark suite (20+ episodes, multiple models) takes 30-60 minutes and is too slow for PR validation. Conversely, running only unit tests doesn't catch quality regressions in the AI pipeline.

## Decision

We adopt a **Multi-Tiered Benchmarking Strategy**:

1. **Smoke Tests (PR Tier)**: Runs on every Pull Request. Uses a tiny, representative subset (3 episodes) and a single baseline config. Goal: Catch total pipeline breakages in <5 minutes.
2. **Full Benchmarks (Nightly Tier)**: Runs nightly on `main`. Uses the full dataset (20+ episodes) and multiple "stress case" configurations. Goal: Detect subtle quality or latency regressions.

## Rationale

- **Developer Velocity**: Smoke tests provide near-instant feedback without bottlenecking the PR queue.
- **Thoroughness**: Nightly runs ensure we don't miss long-term drift that only appears across a larger data sample.
- **Cost Control**: Avoids expensive API calls or massive GPU usage on every minor commit push.

## Alternatives Considered

1. **Full Benchmarks on PR**: Rejected as it kills developer momentum.
2. **No Benchmarks in CI**: Rejected as it allows silent quality regressions to reach production.

## Consequences

- **Positive**: High CI stability; fast feedback loop; comprehensive nightly coverage.
- **Negative**: Requires maintaining two separate CI workflows.

## References

- [RFC-041: Podcast ML Benchmarking Framework](../rfc/RFC-041-podcast-ml-benchmarking-framework.md)
