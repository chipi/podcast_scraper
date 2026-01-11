# ADR-021: Standardized Test Pyramid

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-018](../rfc/RFC-018-test-structure-reorganization.md), [RFC-024](../rfc/RFC-024-test-execution-optimization.md)

## Context & Problem Statement

As the project grew, tests became a mix of unit, integration, and E2E logic in single files. This made it impossible to run "fast" tests only or to isolate why a failure occurred.

## Decision

We enforce a **Standardized Test Pyramid**:

1. **Unit Tests (`tests/unit/`)**: Pure logic, zero IO, sub-second execution.
2. **Integration Tests (`tests/integration/`)**: Module interactions, filesystem tests, mock-API calls.
3. **End-to-End Tests (`tests/e2e/`)**: Full CLI runs, real ML model loading, local server mocks.

Tests are further categorized using **Pytest Markers** (`@pytest.mark.slow`, `@pytest.mark.ml_models`).

## Rationale

- **Feedback Speed**: Developers can run `make test-unit` in seconds during the inner loop.
- **Reliability**: Isolated tests make it clear whether a bug is in a specific function or an integration point.
- **CI Control**: Enables the "Stratified CI" (ADR-017) by providing clear targets for fast/full checks.

## Alternatives Considered

1. **Feature-based Organization**: Rejected as it makes it harder to separate fast vs. slow tests.

## Consequences

- **Positive**: Highly predictable test suite; clear contribution guidelines for new tests.
- **Negative**: Requires strict discipline to keep IO out of unit tests.

## References

- [RFC-018: Test Structure Reorganization](../rfc/RFC-018-test-structure-reorganization.md)
- [RFC-024: Test Execution Optimization](../rfc/RFC-024-test-execution-optimization.md)
