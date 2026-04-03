# ADR-040: Materialization Boundary for Evaluation Inputs

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-046](../rfc/RFC-046-materialization-architecture.md)

## Context & Problem Statement

Preprocessing profiles (e.g. `cleaning_v3` vs `cleaning_v4`) were specified in run
configs alongside model parameters. Two runs with different preprocessing but the same
`dataset_id` appeared comparable when they were not — the v7 experiment showed an 80
percentage-point improvement from preprocessing alone. This made evaluation comparisons
ambiguous and input contracts implicit.

The boundary between "what is the dataset?" and "what does this run configure?" needed
a principled definition.

## Decision

Preprocessing is moved from a run-time parameter to a **dataset materialization**
parameter. The materialization boundary is defined as follows:

1. **Materialization owns preprocessing**: The combination of
   `(dataset_id + canonical_profile + adapter)` produces a `materialization_id`. Runs
   reference a `materialization_id`, not a raw `dataset_id` with a separate
   `preprocessing_profile`.
2. **Two-layer preprocessing model**: Layer A (canonical cleanup) is shared across all
   providers. Layer B (adapter) is provider-specific (e.g. speaker anonymization for
   ML models). Both are applied at materialization time, not run time.
3. **Chunking stays in run config**: Chunking is model-dependent (BART needs 1024
   tokens, LED handles 4096). It belongs in run config because materializing chunks
   would lock inputs to one model's requirements.
4. **Semantic versioning in materialization ID**: Materialization configs carry semver
   (`version: "1.0.0"`) so materialized inputs can evolve while preserving historical
   datasets.
5. **Materialized inputs are frozen**: Once generated, materialized text files are
   immutable. Changes produce a new version.

## Rationale

- The v7 experiment proved that preprocessing is not a minor parameter — it changes
  *what the input is*, not *how the model processes it*. It belongs in the dataset
  definition.
- Separating canonical from adapter preprocessing allows fair cross-provider
  comparison (canonical only) while enabling provider-specific optimization (with
  adapter).
- Chunking is the one preprocessing step that genuinely depends on the model's context
  window, so it stays in run config.
- Semver enables reproducibility: a specific `materialization_id` always produces
  identical inputs.

## Alternatives Considered

1. **Keep preprocessing as run parameter (status quo)**: Rejected; ambiguous
   comparisons and hidden input differences (proven by v7 experiment).
2. **Include chunking in materialization**: Rejected; model-dependent — would require
   separate materializations per chunk size, defeating the purpose of a shared input
   contract.
3. **Apply adapters at runtime instead of materialization time**: Rejected; makes
   adapters a hidden run parameter, which is the problem this ADR solves.

## Consequences

- **Positive**: Comparisons are honest — same `materialization_id` guarantees same
  inputs. Materialized datasets are frozen and auditable. Provider-specific optimization
  is explicit.
- **Negative**: Adds a materialization generation step before experiments. Requires
  migration from `preprocessing_profile` in existing configs.
- **Neutral**: `data/eval/materializations/` and `data/eval/materialized/` directories
  are added to the project structure.

## Implementation Notes

- **Module**: `podcast_scraper/evaluation/` — materialization config loading and
  generation
- **Config**: `data/eval/materializations/*.yaml` — materialization definitions
- **Output**: `data/eval/materialized/<materialization_id>/` — frozen text files
- **Migration**: `preprocessing_profile` in run configs is deprecated; backward
  compatibility layer logs a warning during transition

## References

- [RFC-046: Materialization Architecture](../rfc/RFC-046-materialization-architecture.md)
- [ADR-013: Standalone Experiment Configuration](ADR-013-standalone-experiment-configuration.md)
- [ADR-017: Registered Preprocessing Profiles](ADR-017-registered-preprocessing-profiles.md)
