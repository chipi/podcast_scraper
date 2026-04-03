# ADR-051: Single Code Path for Evaluation and Application

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-048](../rfc/RFC-048-evaluation-application-alignment.md)

## Context & Problem Statement

Evaluation runs and application (production/dev) runs used the same models but could
diverge in behavior due to implicit defaults, separate eval-only logic, and hidden
parameter drift. When evaluation metrics looked good but production behavior differed,
root-cause analysis was difficult because the two paths were not guaranteed to share the
same execution logic.

The core question: should evaluation exercise a separate code path optimized for
measurement, or the exact same code path that ships to users?

## Decision

Evaluation and application **share a single execution path**. No separate "eval
summarizer" or "prod summarizer" exists.

1. **One code path**: Eval runs call the same providers, pipeline stages, and
   preprocessing logic as the application. Shared provider initialization, shared
   generation logic, shared dynamic safeguards.
2. **Explicit parameters only**: All behavioral ML parameters (`max_new_tokens`,
   `min_new_tokens`, `early_stopping`, `no_repeat_ngram_size`, chunking strategy) must
   appear explicitly in config. No silent defaults for parameters that affect output.
3. **Scorers are read-only observers**: Scorers never mutate behavior. They operate on
   predictions *after* generation and never change generation parameters, filter
   predictions, or affect chunking/reduce decisions. All filtering (e.g. NER scope
   filtering) happens inside the scorer.
4. **Preprocessing is part of the model contract**: The preprocessing profile is
   included in the fingerprint and treated as a behavioral parameter, not an
   incidental setting.
5. **Dynamic safeguards run everywhere**: Runtime safety logic (capping
   `max_new_tokens` based on input size, forcing `min_new_tokens=0` to prevent
   expansion) runs in both eval and app. These are provider responsibilities, not
   eval-only features.

## Rationale

- If eval and app diverge, metrics are misleading — "what you evaluate is what you
  ship" is the non-negotiable principle.
- Explicit parameters eliminate hidden drift. When behavior changes, it is traceable to
  a config change, not an implicit default shift.
- Read-only scorers guarantee that adding or modifying evaluation logic never introduces
  production regressions.
- Including preprocessing in the fingerprint means two runs with different cleaning
  profiles are correctly identified as different, not silently compared.

## Alternatives Considered

1. **Separate eval mode with stricter constraints**: Rejected; creates two paths that
   can diverge. "Stricter" eval-only policies make eval results non-representative of
   production.
2. **Shared code with eval-only parameter overrides**: Rejected; overrides reintroduce
   hidden divergence. If an override matters for eval, it matters for production too.
3. **Allow implicit defaults for non-critical parameters**: Rejected; difficult to
   determine what is "non-critical." The v7 experiment showed preprocessing — seemingly
   minor — had an 80% impact.

## Consequences

- **Positive**: Eval results are trustworthy indicators of production behavior. All
  behavior is explainable from logs + config. No hidden drift.
- **Negative**: Cannot add eval-only optimizations (e.g. eval-fast presets) without
  ensuring they also work in production. Slightly more verbose configs (every parameter
  explicit).
- **Neutral**: Fingerprints become the source of truth for explaining and comparing
  runs.

## Implementation Notes

- **Module**: `src/podcast_scraper/evaluation/`, `src/podcast_scraper/providers/ml/`
- **Pattern**: Scorers implement a read-only observer interface; providers own dynamic
  safeguards
- **Artifact contract**: Every run (eval or app) produces `predictions.jsonl`,
  `metrics.json`, `metrics_report.md`, `fingerprint.json`, and `run.log`
- **Relationship to ADR-040**: Materialization (ADR-040) ensures inputs are identical;
  this ADR ensures execution is identical

## References

- [RFC-048: Evaluation-Application Alignment](../rfc/RFC-048-evaluation-application-alignment.md)
- [ADR-049: Materialization Boundary](ADR-049-materialization-boundary-for-eval-inputs.md)
- [ADR-015: Deep Provider Fingerprinting](ADR-015-deep-provider-fingerprinting.md)
- [ADR-017: Registered Preprocessing Profiles](ADR-017-registered-preprocessing-profiles.md)
