# ADR-040: Periodic Module Coupling Analysis

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-038](../rfc/RFC-038-continuous-review-tooling.md)

## Context & Problem Statement

As a project grows, modules often become "tangled," leading to circular imports, high fan-out (one module importing 20 others), and a "big ball of mud" architecture. These issues are hard to see until they cause a runtime crash.

## Decision

We implement **Periodic Module Coupling Analysis**.

- Use `pydeps` to generate visual dependency graphs.
- Integrated into the **Nightly CI** and accessible via `make deps-graph`.
- Enforce architectural thresholds:
  - **Zero Circular Imports** allowed.
  - **Max Dependency Depth** of 5.
  - **Max Imports per Module** of 15.

## Rationale

- **Visibility**: Makes architectural decay visible through daily-updated diagrams.
- **Prevention**: Thresholds act as "quality gates" that alert maintainers before a module becomes too complex to refactor.
- **Onboarding**: Visual graphs help new contributors understand the project's module boundaries.

## Alternatives Considered

1. **Manual Review**: Rejected as humans are poor at tracing large-scale import graphs.

## Consequences

- **Positive**: Clean, modular architecture; no circular import "surprises."
- **Negative**: Requires maintaining `pydeps` and its system dependencies (Graphviz) in CI.

## References

- [RFC-038: Continuous Review Tooling Implementation](../rfc/RFC-038-continuous-review-tooling.md)
