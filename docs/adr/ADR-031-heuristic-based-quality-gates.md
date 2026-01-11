# ADR-031: Heuristic-Based Quality Gates

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md)

## Context & Problem Statement

Standard NLP metrics like ROUGE or BLEU are excellent for general summarization but fail to catch podcast-specific failures, such as a model leaking speaker labels ("Speaker 1: ...") or boilerplate text ("Credits: ...") into the final summary.

## Decision

We implement **Heuristic-Based Quality Gates**.

- In addition to ROUGE, the evaluation pipeline runs fast, regex-based "heuristic" checks.
- Gates include: `boilerplate_leak_rate`, `speaker_label_leak_rate`, `repetition_score`, and `ellipsis_rate` (truncation detection).
- Critical gates (like zero-tolerance for speaker leaks) can trigger "major" alerts even if ROUGE scores are high.

## Rationale

- **Precision**: Catches 80% of the "eyeball regressions" that developers usually find manually.
- **Speed**: These checks are pure Python/regex and complete in milliseconds.
- **Actionability**: Provides specific feedback (e.g., "Regressed: 2 instances of speaker labels found") rather than just a lower score.

## Alternatives Considered

1. **LLM-as-a-Judge**: Considered but rejected for v1 due to cost and latency. Heuristics catch the most common failures more cheaply.

## Consequences

- **Positive**: Higher summary quality; automated detection of known podcast failure modes.
- **Negative**: Requires tuning regex patterns to avoid false positives.

## References

- [RFC-041: Podcast ML Benchmarking Framework](../rfc/RFC-041-podcast-ml-benchmarking-framework.md)
