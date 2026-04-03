# ADR-020: Acceptance Test Tier as Final CI Gate

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-023](../rfc/RFC-023-readme-acceptance-tests.md)

## Context & Problem Statement

The project's test pyramid (ADR-020) defines three tiers: unit, integration, and E2E.
E2E tests verify **feature completeness** — "does the feature work?" — but nothing
validates whether the README's installation commands, CLI examples, and feature claims
are actually accurate. Documentation can drift silently from reality, and first-time
users encounter broken examples with no automated detection.

A new tier is needed that sits **after** all other tests and tests
**documentation accuracy**, not feature completeness.

## Decision

We introduce a fourth test tier: **acceptance tests**.

1. **Location**: `tests/acceptance/` — a new top-level directory, distinct from
   `tests/e2e/`.
2. **Marker**: `@pytest.mark.acceptance` — separate from `e2e`, `integration`, and
   `slow`.
3. **Purpose**: Verify that every executable example in the README works as documented.
   Tests are derived directly from README content and run the exact commands a new user
   would run.
4. **CI role**: Acceptance tests are the **final CI gate**. They run only after unit,
   integration, and E2E tests all pass. They are allowed to be slow (10–20 minutes).
5. **Execution**: Sequential (not parallelized), on main branch merges and
   `workflow_dispatch` only by default.
6. **Makefile target**: `make test-acceptance`.

## Rationale

- **ADR-020** covers the classic unit/integration/E2E pyramid; none of those tiers test
  "does the README example actually work?" — that is a different question.
- Documentation accuracy is a first-class quality property. Broken README examples erode
  user trust more than a failing internal test.
- Running acceptance tests last avoids wasting time on slow doc-verification when fast
  tests already fail.
- The `acceptance` marker lets developers skip these locally while CI enforces them on
  merge.

## Alternatives Considered

1. **Fold into E2E tests**: Rejected; E2E tests verify features, not documentation. The
   purposes are different, and they need different CI triggers and tolerances.
2. **Manual README verification before release**: Rejected; error-prone and doesn't
   scale. Manual checks are forgotten under time pressure.
3. **Documentation linting only (markdownlint, link checkers)**: Rejected; catches
   formatting issues but not "does the code example actually run?"

## Consequences

- **Positive**: README examples are continuously verified. Documentation drift is caught
  automatically. Users can trust that README commands work.
- **Negative**: Adds a slow CI job (10–20 min). README changes require corresponding
  acceptance test updates. Total CI time increases.
- **Neutral**: A new `tests/acceptance/` directory and `acceptance` pytest marker are
  added to the project structure.

## Implementation Notes

- **Module**: `tests/acceptance/`
- **Pattern**: Tests use `subprocess` to run exact README commands against E2E server
  fixtures (no external network).
- **CI**: `test-acceptance` job depends on all other test jobs passing.
- **Relationship to ADR-020**: Extends the three-tier pyramid with a fourth
  documentation-accuracy tier; does not replace any existing tier.

## References

- [RFC-023: README Acceptance Tests](../rfc/RFC-023-readme-acceptance-tests.md)
- [ADR-019: Standardized Test Pyramid](ADR-019-standardized-test-pyramid.md)
- [Testing Strategy](../TESTING_STRATEGY.md)
