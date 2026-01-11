# ADR-020: Linear History via Squash-Merge

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md)

## Context & Problem Statement

Standard merge commits create a "braided" history that makes it difficult to pinpoint where a bug was introduced or to revert a feature cleanly. Feature branches often contain dozens of "fix typo" or "checkpoint" commits that add noise to the main history.

## Decision

We mandate a **Rebase-and-Squash Workflow**:

1. All Pull Requests MUST be squashed into a single commit when merging to `main`.
2. Merge commits are disabled in repository settings.
3. The merge commit message should reference the PR and any linked issues.

## Rationale

- **Readability**: The `main` branch history remains a clean list of completed features and fixes.
- **Revertibility**: A feature can be fully reverted by undoing a single commit.
- **Bisectability**: `git bisect` is much faster and more reliable on linear histories.

## Alternatives Considered

1. **Standard Merge**: Rejected due to history noise.
2. **Rebase Merge (without squash)**: Rejected as it preserves small, unhelpful intermediate commits.

## Consequences

- **Positive**: Professional, readable history; easier to generate changelogs.
- **Negative**: Intermediate development history is lost on the remote (though preserved locally if worktrees are kept).

## References

- [RFC-039: Development Workflow with Git Worktrees and CI Evolution](../rfc/RFC-039-development-workflow-worktrees-ci.md)
