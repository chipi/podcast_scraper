# ADR-017: Stratified CI Execution

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md)

## Context & Problem Statement

Running the full suite of integration and E2E tests (~10+ minutes) on every single commit push is wasteful and slows down the development cycle. However, skipping them entirely until merge is risky.

## Decision

We implement **Stratified CI Execution**:

1. **Push Trigger (Fast Checks)**: On every push to a feature branch, run only unit tests, linting, and formatting (~2 minutes).
2. **Pull Request Trigger (Full Validation)**: When a PR is opened or updated, run the entire suite (Integration, E2E, Coverage, Docs) (~10 minutes).

## Rationale

- **Velocity**: Developers get near-instant feedback on basic regressions.
- **Resource Efficiency**: Saves GitHub Actions minutes by not running heavy ML-simulated tests on small iterative commits.
- **Safety**: The "merge gate" (PR) still ensures 100% validation before code hits `main`.

## Alternatives Considered

1. **Full Suite on Every Push**: Rejected as it bottlenecks the "Commit-Push-Verify" loop.
2. **Manual CI**: Rejected as it relies on human memory and is error-prone.

## Consequences

- **Positive**: Faster development iterations; lower CI costs.
- **Negative**: Integration errors may not be discovered until the PR stage.

## References

- [RFC-039: Development Workflow with Git Worktrees and CI Evolution](../rfc/RFC-039-development-workflow-worktrees-ci.md)
