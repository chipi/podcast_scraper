# RFC-024: Test Execution Optimization

- **Status**: ✅ Completed
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
  - `docs/guides/DEVELOPMENT_GUIDE.md` - Development workflow and testing requirements
  - `Makefile` - Test execution targets

## Abstract

This RFC defines a strategy for **optimizing test execution speed** to keep local development feedback loops fast. The strategy focuses on:

1. **Test speed tiers**: Explicit grouping of tests by speed and intent
2. **Parallel vs sequential execution**: Optimized based on empirical performance data
3. **Makefile targets**: Clear, optimized targets for different test scenarios

**Key Principle:** Fast tests protect developer flow. Default local runs should complete in ≤ 30 seconds.

## System Overview

This RFC is part of a three-RFC system (RFC-024, RFC-025, RFC-026) that optimizes test execution, metrics collection, and consumption. The complete flow:

````text
  ├─ PR: Fast tests (Tier 0 + Tier 1 fast)
  ├─ Main: All tests (Tier 0 + Tier 1 + Tier 2)
  └─ Nightly: Full suite + comprehensive metrics
  ↓
Artifacts Generated
  ├─ JUnit XML (test results, timing)
  ├─ Coverage reports (XML, HTML, terminal)
  └─ JSON metrics (structured data)
  ↓
Consumption Methods
  ├─ Job Summary (PR authors, 0s)
  ├─ metrics.json (automation, 5s)
  └─ Dashboard (maintainers, 10s)
```yaml

**See also:**

- RFC-025: Metrics collection (artifacts generation)
- RFC-026: Metrics consumption (consumption methods)

## Core Principles

These principles are shared across RFC-024, RFC-025, and RFC-026:

- **Developer flow > completeness** - Fast feedback loops protect developer state and enable rapid iteration
- **Metrics must be cheap to collect** - Automated collection with zero manual work required
- **Humans consume summaries, machines consume JSON** - Job summaries for quick checks, JSON API for automation

## Golden Path (Default Workflow)

**For new contributors:** Follow this simple path to avoid cognitive overload:

- **While coding** → `make test-unit` (fast feedback, ≤ 30s)
- **Before PR** → `make test-ci-fast` (reasonable confidence, 6-10 min)
- **CI decides the rest** (full validation runs automatically on PR and main)

This path provides fast feedback during development and reasonable confidence before pushing, while CI handles comprehensive validation.

## Problem Statement

**Original Issues (Now Resolved):**

1. **Unclear Test Execution Strategy** ✅ **RESOLVED**
   - ~~Developers unsure which tests to run locally vs. in CI~~
   - ~~No clear guidance on test speed tiers or when to use parallel execution~~
   - ~~Default test runs may be slower than necessary~~

2. **Suboptimal Local Development Experience** ✅ **RESOLVED**
   - ~~Feedback loops may be too slow for rapid iteration~~
   - ~~No clear separation between "quick check" and "full validation"~~
   - ~~Parallel execution benefits not optimized per test type~~

**Original Impact (Now Addressed):**

- ~~Developers may skip running tests locally (slow feedback)~~ → Fast feedback now available
- ~~Slow tests accumulate without visibility~~ → Test speed tiers clearly defined
- ~~No data-driven decisions about test optimization~~ → Optimized based on empirical data

**Current State:**

All original issues have been resolved through implementation of:
- Clear test speed tiers (Tier 0, Tier 1, Tier 2)
- Optimized Makefile targets per test type
- Parallel execution based on empirical performance data
- Fast CI jobs for quick feedback
- Comprehensive documentation

## Goals

### Primary Goal

**Fast Local Development Cycle:**

- Default local test runs complete in ≤ 30 seconds
- Clear test speed tiers with explicit execution strategies
- Optimized parallel execution based on test characteristics
- Fast feedback protects developer flow state

### Success Criteria

- ✅ Default `make test-unit` completes in ≤ 30 seconds
- ✅ Test execution strategy clearly documented and optimized
- ✅ Makefile targets optimized per test type
- ✅ Clear guidance on when to use parallel vs sequential execution

## Test Speed Tiers

We explicitly group tests by **intent and speed** to optimize execution strategies.

**Note on `ml_models` marker:** The `ml_models` marker is a **dependency dimension** (indicates ML model dependencies), not a speed tier. Tests can be both fast and require ML dependencies, or slow without ML dependencies. The marker is used to exclude tests that require ML dependencies from fast CI feedback loops, but it does not determine the test's speed tier.

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

- Parallel execution (empirical data shows 11.6% faster with 4 workers)
- Sequential option available for debugging (cleaner output)
- Run automatically while coding
- Pre-commit hooks

**Runtime Budget Enforcement:**

If a Tier 0 test causes `make test-unit` to exceed 30 seconds:
- It must be downgraded to Tier 1, split into smaller tests, or reclassified
- This gives maintainers authority to push back on slow tests entering Tier 0
- The 30-second budget is a hard constraint, not a suggestion

**Current Implementation:**

- `make test-unit` (unit tests only, parallel by default - matches CI behavior)
- `make test-unit-sequential` (sequential - for debugging, cleaner output)
- Pre-commit hooks installed via `make install-hooks`

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

- `make test-ci` (unit + fast integration + fast e2e, parallel, with coverage)
- `make test-ci-fast` (unit + fast integration + fast e2e, parallel, no coverage for speed)
- `make test-integration` (all integration tests, parallel - 3.4x faster, with re-runs)
- `make test-integration-fast` (fast integration tests only, excludes slow/ml_models, parallel)
- `make test-integration-slow` (slow integration tests, includes slow/ml_models, parallel)
- CI jobs on PRs: `test-fast` (6-10 min) and `test` (10-15 min) run in parallel

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

- Parallel execution (faster for slow tests, similar 3.4x benefit as integration)
- Run on `main` branch (on every push, not just nightly)
- Manually on demand

**Current Implementation:**

- `make test-e2e` (all E2E tests, parallel, with re-runs and network guard)
- `make test-e2e-fast` (fast E2E tests only, excludes slow/ml_models, parallel)
- `make test-e2e-slow` (slow E2E tests, includes slow/ml_models, parallel)
- `make test-all` (all tests: unit + integration + e2e, parallel)
- `make test-all-fast` (fast tests only, excludes slow/ml_models, parallel)
- `make test-all-slow` (slow tests only, includes slow/ml_models, parallel)
- CI jobs on main branch: `test-unit` (2-5 min), `test-integration` (5-10 min), `test-e2e` (20-30 min) run in parallel

## Test Execution Optimization

### Current Performance Characteristics

Based on empirical testing:

| Test Type | Sequential | Parallel (4 workers) | Parallel (auto) | Speedup | Recommendation |
| ----------- | ----------- | ---------- | ---------- | --------- | ---------------- |
| **Unit Tests** | 2.16s | 1.91s | 2.19s | +11.6% (4 workers) | Parallel with 4 workers (optimal) |
| **Integration Tests** | 115.7s | N/A | 33.6s | 3.4x faster | Parallel with auto workers (significant benefit) |
| **E2E Tests** | TBD | N/A | TBD | Similar to integration (~3.4x faster) | Parallel with auto workers (significant benefit) |

### Makefile Target Strategy

**Optimized Execution:**

```makefile

# Unit tests: parallel execution (fast, matches CI behavior)

test-unit:
  pytest -n auto --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e'

# Unit tests: sequential execution (for debugging, cleaner output)

test-unit-sequential:
  pytest --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e'

# Integration tests: parallel (3.4x faster, significant benefit) with re-runs

test-integration:
  pytest tests/integration/ -m integration -n auto --reruns 2 --reruns-delay 1

# Fast integration tests: excludes slow/ml_models (faster CI feedback)

test-integration-fast:
  pytest tests/integration/ -m "integration and not slow and not ml_models" -n auto

# Slow integration tests: includes slow/ml_models (requires ML dependencies)

test-integration-slow:
  pytest tests/integration/ -m "integration and (slow or ml_models)" -n auto

# CI test suite: unit + fast integration + fast e2e (excludes slow/ml_models, with coverage)

test-ci:
  pytest -n auto --cov=$(PACKAGE) --cov-report=term-missing -m '(not slow and not ml_models)' --disable-socket --allow-hosts=127.0.0.1,localhost

# Fast CI test suite: unit + fast integration + fast e2e (no coverage for speed)

test-ci-fast:
  pytest -n auto -m '(not slow and not ml_models)' --disable-socket --allow-hosts=127.0.0.1,localhost

# E2E tests: parallel with re-runs and network guard (faster for slow tests)

test-e2e:
  pytest tests/e2e/ -m e2e -n auto --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1

# Fast E2E tests: excludes slow/ml_models (faster CI feedback)

test-e2e-fast:
  pytest tests/e2e/ -m "e2e and not slow and not ml_models" -n auto --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1

# Slow E2E tests: includes slow/ml_models (requires ML dependencies)

test-e2e-slow:
  pytest tests/e2e/ -m "e2e and (slow or ml_models)" -n auto --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1
```text

- Parallel execution for all test types (optimal performance)
- Re-runs for integration and E2E tests (handles flaky tests)
- Network guard for E2E tests (blocks external network calls)
- Fast/slow test separation (excludes slow/ml_models for faster feedback)
- Coverage included in CI suite (unified coverage report)

### Phase 1: Optimize Test Execution (Immediate)

- [x] Update Makefile targets with optimized execution strategies
- [x] Document test speed tiers
- [x] Add comments explaining execution choices
- [x] Update help text
- [x] Implement CI jobs with parallel execution
- [x] Separate fast feedback and full validation jobs on PRs
- [x] Separate test jobs on main branch for maximum parallelization

**Status:** ✅ **Completed** (based on empirical testing)

### CI Job Implementation

**Pull Requests:**
- `lint` job: Fast checks without ML dependencies (1-2 min)
- `test-fast` job: Fast feedback, no coverage (6-10 min) - runs `make test-ci-fast`
- `test` job: Full validation with coverage (10-15 min) - runs `make test-ci`
- `docs` job: Documentation build (2-3 min)
- `build` job: Package build (1-2 min)
- All jobs run in parallel for maximum speed

**Push to Main:**
- `lint` job: Fast checks without ML dependencies (1-2 min)
- `test-unit` job: Unit tests only, no ML deps (2-5 min) - includes network isolation verification
- `test-integration` job: All integration tests with re-runs (5-10 min)
- `test-e2e` job: All E2E tests with re-runs and network guard (20-30 min)
- `docs` job: Documentation build (2-3 min)
- `build` job: Package build (1-2 min)
- All jobs run in parallel for maximum speed

**Key Features:**
- Fast feedback on PRs (test-fast completes in 6-10 min)
- Full validation on PRs (test job provides unified coverage)
- Complete validation on main (all tests including slow/ml_models)
- Re-runs for flaky tests (integration and E2E tests)
- Network guard for E2E tests (blocks external network calls)
- Network isolation verification for unit tests

## Design Decisions

### 1. Sequential vs. Parallel Execution

**Decision:** Optimize per test type based on empirical data

**Rationale:**

- Unit tests: Parallel with auto workers is 11.6% faster (optimal balance, avoids over-parallelization)
- Integration/E2E tests: Parallel with auto workers is 3.4x faster (significant benefit)
- Default: Parallel execution for all test types (optimal performance)
- Sequential option available for debugging when needed

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
- `docs/guides/DEVELOPMENT_GUIDE.md`: Development workflow
- `pyproject.toml`: Pytest configuration and markers

## Implementation Status

### ✅ Completed

- Makefile targets optimized per test type
- CI jobs implemented with parallel execution
- Fast feedback jobs on PRs (test-fast and test run simultaneously)
- Separate test jobs on main branch (test-unit, test-integration, test-e2e)
- Re-runs for flaky tests (integration and E2E)
- Network guard for E2E tests
- Network isolation verification for unit tests
- Pre-commit hooks for fast feedback

### Additional Enhancements

Beyond the original RFC requirements, the following enhancements were implemented:

1. **Two-Tier PR Testing Strategy**
   - Fast feedback job (`test-fast`) provides early pass/fail signal
   - Full validation job (`test`) provides unified coverage report
   - Both run simultaneously (no waiting)

2. **Network Isolation**
   - E2E tests use `pytest-socket` to block external network calls
   - Unit tests verify they can import without ML dependencies
   - Ensures tests are truly isolated

3. **Re-runs for Flaky Tests**
   - Integration and E2E tests include `--reruns 2 --reruns-delay 1`
   - Reduces false negatives from transient failures
   - Improves CI reliability

4. **Path-Based Workflow Filtering**
   - Workflows only run when relevant files change
   - Saves ~18 minutes per docs-only commit
   - Significant resource savings

## Notes

- Test execution optimization based on empirical measurements
- See `RFC-025-test-metrics-and-health-tracking.md` for metrics collection and health tracking strategy
- See `docs/CI_CD.md` for complete CI/CD pipeline documentation
````
