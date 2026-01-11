# ADR-018: Isolated Runtime Environments

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md)

## Context & Problem Statement

Sharing a single global virtual environment across different project versions (e.g., `main` needing `transformers` v4.40 and a legacy branch needing v4.30) leads to constant `pip install` churn and subtle bugs.

## Decision

We mandate **Independent Virtual Environments per Worktree**.

- Every worktree directory MUST have its own `.venv/` folder.
- The `Makefile` is designed to detect and use the local environment.

## Rationale

- **Deterministic Dependencies**: Guarantees that the code being edited is running against the exact dependencies defined in its `pyproject.toml`.
- **Side-by-Side Execution**: Allows running two different versions of the scraper simultaneously on the same machine without library conflicts.
- **Clean Cleanup**: Deleting a worktree folder automatically removes its environment.

## Alternatives Considered

1. **Global Conda/Pyenv**: Rejected as it makes it difficult to manage the "multiple versions at once" requirement.

## Consequences

- **Positive**: Rock-solid dependency isolation; zero "works on my machine" issues across branches.
- **Negative**: Increased disk usage (approx. 150MB per environment).

## References

- [RFC-039: Development Workflow with Git Worktrees and CI Evolution](../rfc/RFC-039-development-workflow-worktrees-ci.md)
