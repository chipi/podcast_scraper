# ADR-039: Grouped Dependency Automation

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-038](../rfc/RFC-038-continuous-review-tooling.md)

## Context & Problem Statement

Keeping dependencies updated is critical for security and performance, but automated tools like Dependabot can become overwhelming if they create a separate PR for every minor package update (especially in large ML stacks).

## Decision

We adopt **Grouped Dependency Automation**.

- Use GitHub Dependabot with specific **Grouping Rules**:
  - `dev-tools`: Group all linting, formatting, and testing tools into one PR.
  - `docs`: Group all MkDocs and documentation plugins.
- Frequency is set to **Weekly** (Monday mornings) to align with the development cycle.
- ML-heavy libraries (like `transformers` or `torch`) are limited to **Patch-only** updates to prevent breaking change noise.

## Rationale

- **Review Efficiency**: One PR covering 5 linting tools is far easier to review than 5 individual PRs.
- **Stability**: Prevents "dependency fatigue" and ensures that breaking ML changes are handled with manual oversight.
- **Security**: Ensures that critical patches are still caught and surfaced weekly.

## Alternatives Considered

1. **Standard Dependabot**: Rejected due to high PR volume and noise.
2. **Manual Updates**: Rejected as it inevitably leads to stale, insecure dependencies over time.

## Consequences

- **Positive**: Clean PR queue; consistent weekly maintenance rhythm.
- **Negative**: Grouped PRs can be harder to debug if one of the 5 packages has a regression.

## References

- [RFC-038: Continuous Review Tooling Implementation](../rfc/RFC-038-continuous-review-tooling.md)
