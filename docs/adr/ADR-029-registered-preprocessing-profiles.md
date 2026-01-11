# ADR-029: Registered Preprocessing Profiles

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-016](../rfc/RFC-016-modularization-for-ai-experiments.md)

## Context & Problem Statement

Transcript cleaning (removing ads, timestamps, or fillers) has as much impact on summary quality as the model itself. However, cleaning logic was often "hidden" inside functions, making it hard to track how changes to regex patterns affected overall metrics.

## Decision

We move cleaning logic into **Registered Preprocessing Profiles**.

- Profiles are defined as versioned objects (e.g., `cleaning_v3`).
- Each profile specifies exactly which steps are active (e.g., `remove_sponsors=True`).
- The `profile_id` is recorded in the output fingerprint (ADR-027).

## Rationale

- **Isolating Variables**: Allows researchers to test Model A vs. Model B while keeping the Preprocessing Profile identical.
- **Traceability**: If ROUGE scores improve, we can definitively say if it was due to a better model or a better cleaning profile.
- **Reusability**: Standardizes cleaning across all episodes in a dataset.

## Alternatives Considered

1. **Ad-hoc Cleaning**: Rejected as it makes benchmarking impossible to reproduce or explain.

## Consequences

- **Positive**: Clearer insight into "data vs. model" performance; easy to roll back cleaning regressions.
- **Negative**: Requires maintaining a registry of profiles.

## References

- [RFC-016: Modularization for AI Experiment Pipeline](../rfc/RFC-016-modularization-for-ai-experiments.md)
