# RFC-024: Test Execution Optimization

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, developers
- **Related PRDs**:
  - `docs/prd/PRD-001-transcript-pipeline.md` (core pipeline)
- **Related RFCs**:
  - `docs/rfc/RFC-018-test-structure-reorganization.md` (test structure - foundation)
  - `docs/rfc/RFC-019-e2e-test-improvements.md` (E2E test infrastructure)
  - `docs/rfc/RFC-020-integration-test-improvements.md` (integration test improvements)
  - `docs/rfc/RFC-023-readme-acceptance-tests.md` (acceptance tests)
  - `docs/rfc/RFC-025-test-metrics-and-health-tracking.md` (test metrics - complementary)
- **Related Documents**:
  - `docs/TESTING_STRATEGY.md` - Overall testing strategy and test categories
  - `docs/DEVELOPMENT_GUIDE.md` - Development workflow and testing requirements
  - `Makefile` - Test execution targets

## Abstract

This RFC defines a strategy for **optimizing test execution speed** to keep local development feedback loops fast. The strategy focuses on:

1. **Test speed tiers**: Explicit grouping of tests by speed and intent
2. **Parallel vs sequential execution**: Optimized based on empirical performance data
3. **Makefile targets**: Clear, optimized targets for different test scenarios

**Key Principle:** Fast tests protect developer flow. Default local runs should complete in ≤ 30 seconds.

## Problem Statement

**Current Issues:**

1. **Unclear Test Execution Strategy**
   - Developers unsure which tests to run locally vs. in CI
   - No clear guidance on test speed tiers or when to use parallel execution
   - Default test runs may be slower than necessary

2. **Suboptimal Local Development Experience**
   - Feedback loops may be too slow for rapid iteration
   - No clear separation between "quick check" and "full validation"
   - Parallel execution benefits not optimized per test type

**Impact:**

- Developers may skip running tests locally (slow feedback)
- Slow tests accumulate without visibility
- No data-driven decisions about test optimization

## Goals

### Primary Goal

**Fast Local Development Cycle:**

- Default local test runs complete in ≤ 30 seconds
- Clear test speed tiers with explicit execution strategies
- Optimized parallel execution based on test characteristics
- Fast feedback protects developer flow state

### Success Criteria

- ✅ Default `make test` completes in ≤ 30 seconds
- ✅ Test execution strategy clearly documented and optimized
- ✅ Makefile targets optimized per test type
- ✅ Clear guidance on when to use parallel vs sequential execution

## Test Speed Tiers

We explicitly group tests by **intent and speed** to optimize execution strategies.

### Tier 0 – Ultra-fast (Default Local)

**Target runtime:** ≤ 10–30 seconds
**Purpose:** Immediate feedback while coding

**Characteristics:**

- Pure unit tests
- No network access
- No real filesystem I/O (except tempfile operations)
- No database
- No sleeps or timeouts
- Fully mocked external dependencies

**Execution Strategy:**

- Sequential execution (overhead dominates for fast tests)
- Run automatically while coding
- Pre-commit hooks
- Watch mode on file changes

**Current Implementation:**

- `make test` (unit tests only, parallel by default)
- `make test-unit` (sequential - faster for fast tests)
- `make test-sequential` (for debugging)

### Tier 1 – Fast Confidence

**Target runtime:** ≤ 1–3 minutes
**Purpose:** Reasonable confidence before pushing

**Characteristics:**

- Unit tests + lightweight integration tests
- Local or in-memory dependencies
- Mocked external services
- Deterministic external interactions

**Execution Strategy:**

- Parallel execution (significant speedup for slower tests)
- Run before pushing to PR
- Always run on pull requests in CI

**Current Implementation:**

- `make test-ci` (unit + integration, parallel)
- `make test-integration` (parallel - 3.4x faster)

### Tier 2 – Full Validation

**Target runtime:** 10+ minutes
**Purpose:** End-to-end system confidence

**Characteristics:**

- Full integration tests
- End-to-end (E2E) tests
- Realistic service wiring
- Cross-component flows
- Real ML models (when appropriate)

**Execution Strategy:**

- Parallel execution (faster for slow tests)
- Run on `main` branch
- Nightly CI
- Manually on demand

**Current Implementation:**

- `make test-workflow-e2e` (parallel - faster for slow tests)
- CI runs on main branch only

## Test Execution Optimization

### Current Performance Characteristics

Based on empirical testing:

| Test Type | Sequential | Parallel | Speedup | Recommendation |
| ----------- | ----------- | ---------- | --------- | ---------------- |
| **Unit Tests** | 2.1s | 2.5s | -20% (slower) | Sequential (overhead dominates) |
| **Integration Tests** | 115.7s | 33.6s | 3.4x faster | Parallel (significant benefit) |
| **E2E Tests** | TBD | TBD | Expected similar to integration | Parallel (expected benefit) |

### Makefile Target Strategy

**Optimized Execution:**

```makefile
# Default: parallel (matches CI, reasonable for most cases)
test:
  pytest -n auto --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not workflow_e2e and not network'

# Sequential: for debugging (cleaner output)
test-sequential:
  pytest --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not workflow_e2e and not network'

# Unit tests: sequential (faster for fast tests, overhead dominates in parallel)
test-unit:
  pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not workflow_e2e and not network'

# Integration tests: parallel (3.4x faster, significant benefit)
test-integration:
  pytest tests/integration/ -m integration -n auto

# E2E tests: parallel (faster for slow tests, similar to integration)
test-workflow-e2e:
  pytest tests/workflow_e2e/ -m workflow_e2e -n auto
```

## Implementation Plan

### Phase 1: Optimize Test Execution (Immediate)

- [x] Update Makefile targets with optimized execution strategies
- [x] Document test speed tiers
- [x] Add comments explaining execution choices
- [x] Update help text

**Status:** ✅ **Completed** (based on empirical testing)

## Design Decisions

### 1. Sequential vs. Parallel Execution

**Decision:** Optimize per test type based on empirical data

**Rationale:**

- Unit tests: Sequential is faster (overhead dominates)
- Integration/E2E tests: Parallel is 3.4x faster (significant benefit)
- Default: Parallel (matches CI, reasonable for most cases)

### 2. Test Speed Tiers

**Decision:** Explicit three-tier system (Tier 0, Tier 1, Tier 2)

**Rationale:**

- Clear guidance on what to run when
- Optimized execution per tier
- Matches CI strategy

## Benefits

### Developer Experience

- ✅ **Fast feedback**: Default runs complete in ≤ 30 seconds
- ✅ **Clear guidance**: Obvious what to run, when, and why
- ✅ **Optimized execution**: Best performance per test type
- ✅ **Debugging support**: Sequential option available when needed

## Related Files

- `Makefile`: Test execution targets (optimized per test type)
- `.github/workflows/python-app.yml`: CI test jobs
- `docs/TESTING_STRATEGY.md`: Overall testing strategy
- `docs/DEVELOPMENT_GUIDE.md`: Development workflow
- `pyproject.toml`: Pytest configuration and markers

## Notes

- Test execution optimization based on empirical measurements
- See `RFC-025-test-metrics-and-health-tracking.md` for metrics collection and health tracking strategy
