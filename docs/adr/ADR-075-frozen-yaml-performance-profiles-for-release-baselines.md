# ADR-075: Frozen YAML Performance Profiles for Release Resource Baselines

- **Status**: Accepted
- **Date**: 2026-04-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md)
- **Related PRDs**: [PRD-016](../prd/PRD-016-operational-observability-pipeline-intelligence.md),
  [PRD-007](../prd/PRD-007-ai-quality-experiment-platform.md)

## Context & Problem Statement

[ADR-014](ADR-014-codified-comparison-baselines.md) and [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md)
freeze **quality** comparables. Existing timing hooks ([ADR-027](ADR-027-unified-provider-metrics-contract.md),
`workflow.metrics`, acceptance reports) do not produce a **single, versioned, git-committable
artifact** for **peak memory (RSS), CPU utilization, and wall time per pipeline stage** comparable
across **releases** under controlled conditions.

Without a decision, teams mix ad hoc `psutil` scripts, CI HTML metrics, and eval-session timings —
none of which answer “did **resource cost** regress between vX and vY?” the same way every time.

## Decision

1. **Frozen profiles live under `data/profiles/`** as **YAML** files, committed at release (or
   retroactively captured for past tags), keyed by **release tag** and methodology described in
   [Performance Profile Guide](../guides/PERFORMANCE_PROFILE_GUIDE.md).
2. **Capture and diff are first-class scripts**: `scripts/eval/profile/freeze_profile.py` produces
   profiles; `scripts/eval/profile/diff_profiles.py` (and Makefile **`profile-freeze`** /
   **`profile-diff`**) compare two profiles for terminal workflows.
3. **Schema and fields** are owned by [RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md)
   and the guide — ADR does not duplicate every field; it ratifies the **pattern** (frozen YAML +
   scripts + committed baselines).
4. **Live runtime dashboards** during a run ([RFC-065](../rfc/RFC-065-live-pipeline-monitor.md)) and
   **Streamlit multi-release views** ([RFC-066](../rfc/RFC-066-run-compare-performance-tab.md)) are
   **consumers** of these artifacts, not alternate sources of truth.

## Rationale

- **Parity with quality baselines**: Resource regressions deserve the same discipline as ROUGE or
  gate regressions.
- **Git as audit trail**: Committed YAML makes profile changes reviewable and ties them to release
  tags.
- **Reuse**: Builds on `Metrics`, `EpisodeStageTimings`, and provider fingerprinting rather than a
   new telemetry silo ([RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md) Problem
   Statement).

## Alternatives Considered

1. **Only CI-generated HTML/metrics JSON**: Rejected; not normalized for release-to-release
   pipeline-stage RSS/CPU comparison.
2. **Store profiles only in S3 or external DB**: Rejected for default workflow; complicates offline
   dev and `make profile-diff`.
3. **Merge resource baselines into eval run directories only**: Rejected; eval runs and release
   profiles have different lifecycles; RFC-066 joins them by tag explicitly.

## Consequences

- **Positive**: Repeatable release hygiene; clear input for Streamlit Performance tab and terminal
  diff.
- **Negative**: Maintainers must refresh profiles when methodology or reference dataset changes;
   documented in the guide.
- **Neutral**: v1 freeze methodology may start **single-feed**; multi-feed profiling called out as
   follow-on in RFC-064.

## Implementation Notes

- **Paths**: `data/profiles/*.yaml`, `scripts/eval/profile/freeze_profile.py`,
  `scripts/eval/profile/diff_profiles.py`, Makefile targets **`profile-freeze`**, **`profile-diff`**.
- **Consumer**: `tools/run_compare/` discovers profiles by release tag
  ([RFC-066](../rfc/RFC-066-run-compare-performance-tab.md)).

## References

- [RFC-064: Performance profiling and release freeze](../rfc/RFC-064-performance-profiling-release-freeze.md)
- [Performance Profile Guide](../guides/PERFORMANCE_PROFILE_GUIDE.md)
- [ADR-014: Codified comparison baselines](ADR-014-codified-comparison-baselines.md)
- [ADR-027: Unified provider metrics contract](ADR-027-unified-provider-metrics-contract.md)
