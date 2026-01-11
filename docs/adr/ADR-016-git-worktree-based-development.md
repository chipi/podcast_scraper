# ADR-016: Git Worktree-Based Development

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md)

## Context & Problem Statement

Parallel development on different versions (e.g., hotfixing 2.3 while building 2.5) in a single directory causes constant branch switching, stashing, and "context drift" for AI tools like Cursor.

## Decision

We adopt **Git Worktrees** as the primary development workflow.

- Each major version or feature branch is checked out into a separate physical directory (e.g., `podcast_scraper-next-2.4/`).
- Developers maintain a "Base" reference folder and active "Task" folders.

## Rationale

- **AI Isolation**: Cursor instances see only the relevant code for the active task, preventing AI hallucinations based on stale files.
- **Parallel Stabilization**: Allows running long-running tests on a release branch in one window while coding features in another.
- **Zero-Stash Workflow**: No need to `git stash` when interrupted; just move to the other directory.

## Alternatives Considered

1. **Single Directory Branching**: Rejected as it breaks IDE state and AI context during version jumps.
2. **Multiple Clones**: Rejected because it wastes disk space and doesn't share Git object history.

## Consequences

- **Positive**: High developer velocity; stable AI context; safer parallel work.
- **Negative**: Requires slightly more disk space (~200MB per worktree) and a small learning curve for the `wt-*` Makefile commands.

## References

- [RFC-039: Development Workflow with Git Worktrees and CI Evolution](../rfc/RFC-039-development-workflow-worktrees-ci.md)
