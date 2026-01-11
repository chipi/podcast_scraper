# ADR-026: Explicit Golden Dataset Versioning

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md)
- **Related PRDs**: [PRD-007](../prd/PRD-007-ai-quality-experiment-platform.md)

## Context & Problem Statement

Golden data (human-verified transcripts or high-quality summaries) is the source of truth for evaluation. If golden data is regenerated or modified silently, all historical benchmarks become invalid.

## Decision

We enforce **Explicit Golden Dataset Versioning**.

- Golden datasets are stored in versioned folders (e.g., `data/eval/golden/indicator_v1/`).
- Creation or updates to a golden dataset require a separate, manual-approval pipeline (`make golden`).
- Once a version is tagged and used in a baseline, it is "frozen" and never modified.

## Rationale

- **Stability**: Guarantees that benchmark scores are comparable over months of development.
- **Rigor**: Prevents "gaming the metrics" by accidentally updating the golden reference to match a specific model's output.
- **Auditability**: We always know exactly which version of ground truth was used for a specific project release.

## Alternatives Considered

1. **Live Golden Data**: Rejected as it invalidates historical comparisons whenever a typo is fixed in a ground-truth transcript.

## Consequences

- **Positive**: Rock-solid evaluation foundation; clear version history for ground truth.
- **Negative**: Requires managing multiple versions of evaluation data.

## References

- [RFC-041: Podcast ML Benchmarking Framework](../rfc/RFC-041-podcast-ml-benchmarking-framework.md)
