# ADR-025: Codified Comparison Baselines

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-015](../rfc/RFC-015-ai-experiment-pipeline.md), [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md)
- **Related PRDs**: [PRD-007](../prd/PRD-007-ai-quality-experiment-platform.md)

## Context & Problem Statement

When evaluating a new model or prompt, "better" is often subjective. Without a stable, codified baseline, it is impossible to determine if a change is a genuine improvement or a regression in disguise.

## Decision

We mandate **Codified Comparison Baselines**.

- Every experiment and benchmark MUST reference a specific `baseline_id` (e.g., `bart_led_baseline_v2`).
- A baseline is a frozen artifact directory containing metadata, predictions, and calculated metrics.
- The system prevents comparisons between experiments using different datasets or mismatched baselines.

## Rationale

- **Objectivity**: Moves from "this looks better" to "this improved ROUGE-L by 5% and reduced latency by 20% vs. the baseline."
- **Integrity**: Enforcing `baseline_id` ensures that developers are always comparing "apples to apples."
- **Regression Detection**: Enables automated CI checks that fail if a PR drops below the established baseline.

## Alternatives Considered

1. **Ad-hoc Comparison**: Rejected as it leads to "baseline drift" where improvements are measured against outdated or unknown states.

## Consequences

- **Positive**: Clear regression signals; data-driven decision making; stable project quality targets.
- **Negative**: Requires a one-time effort to create and "freeze" baseline artifacts.

## References

- [RFC-015: AI Experiment Pipeline](../rfc/RFC-015-ai-experiment-pipeline.md)
- [RFC-041: Podcast ML Benchmarking Framework](../rfc/RFC-041-podcast-ml-benchmarking-framework.md)
