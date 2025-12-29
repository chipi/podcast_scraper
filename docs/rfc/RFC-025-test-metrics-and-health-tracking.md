# RFC-025: Test Metrics and Health Tracking

- **Status**: Accepted
- **Authors**:
- **Stakeholders**: Maintainers, developers, CI/CD pipeline maintainers
- **Related PRDs**:
  - `docs/prd/PRD-001-transcript-pipeline.md` (core pipeline)
- **Related RFCs**:
  - `docs/rfc/RFC-018-test-structure-reorganization.md` (test structure - foundation)
  - `docs/rfc/RFC-019-e2e-test-improvements.md` (E2E test infrastructure)
  - `docs/rfc/RFC-020-integration-test-improvements.md` (integration test improvements)
  - `docs/rfc/RFC-023-readme-acceptance-tests.md` (acceptance tests)
  - `docs/rfc/RFC-024-test-execution-optimization.md` (test execution - complementary)
  - `docs/rfc/RFC-026-metrics-consumption-and-dashboards.md` (metrics consumption - complementary)
- **Related Documents**:
  - `docs/TESTING_STRATEGY.md` - Overall testing strategy and test categories
  - `docs/guides/DEVELOPMENT_GUIDE.md` - Development workflow and testing requirements
  - `.github/workflows/python-app.yml` - CI test jobs

## Abstract

This RFC defines a strategy for **tracking test metrics and monitoring codebase health** over time. The strategy focuses on:

1. **Metrics collection**: What to track on every test run
2. **CI integration**: How to collect and display metrics automatically
3. **Trend tracking**: How to monitor health over time
4. **Flaky test detection**: How to identify and track unstable tests

**Key Principle:** Metrics enable improvement. Track runtime, coverage, flakiness, and trends over time to enable data-driven optimization decisions.

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

- RFC-024: Test execution optimization (pytest + markers → CI tiers)
- RFC-026: Metrics consumption (consumption methods)

## Core Principles

These principles are shared across RFC-024, RFC-025, and RFC-026:

- **Developer flow > completeness** - Fast feedback loops protect developer state and enable rapid iteration
- **Metrics must be cheap to collect** - Automated collection with zero manual work required
- **Humans consume summaries, machines consume JSON** - Job summaries for quick checks, JSON API for automation

## Problem Statement

**Current Issues:**

1. **No Systematic Metrics Tracking**
   - No visibility into test runtime trends
   - No tracking of slowest tests over time
   - No historical coverage trends
   - No flaky test identification and tracking

2. **No Data-Driven Optimization**
   - Difficult to identify performance regressions
   - No visibility into which tests are slowing down
   - No historical context for test health

**Impact:**

- Slow tests accumulate without visibility
- No data-driven decisions about test optimization
- Difficult to identify performance regressions
- Flaky tests go undetected

## Goals

### Primary Goal

**Codebase Health Tracking:**

- Monitor key metrics on every test run
- Track trends over time (runtime, coverage, flakiness)
- Identify slow tests and performance regressions
- Enable data-driven optimization decisions

### Success Criteria

- ✅ Metrics automatically collected in CI (no manual work)
- ✅ Historical trends visible (runtime, coverage, flakiness)
- ✅ Slowest tests identified and tracked
- ✅ Flaky tests automatically detected and reported
- ✅ GitHub Actions job summaries display key metrics

## Test Metrics & Monitoring

### Metrics to Track

Track these metrics on every test run to monitor codebase health. Metrics are categorized as **collected** (facts) or **derived** (interpretations) to avoid future debates about what to track.

#### Collected Metrics (Facts)

**1. Runtime Metrics** (Ownership: Test maintainers)

- Total runtime per tier (unit / integration / e2e)
- Individual test runtime
- Top 20 slowest tests

**2. Test Health Metrics** (Ownership: Test maintainers)

- Pass/fail/skip status per test
- Total passed count
- Total failed count
- Total skipped count
- Flaky test count (tests that pass on rerun)

**3. Coverage Metrics** (Ownership: Code owners)

- Overall coverage percentage
- Coverage by module
- Uncovered lines identification

**4. Resource Usage Metrics** (Ownership: CI/infrastructure owners)

- CPU usage
- Memory usage

#### Derived Metrics (Interpretations)

**1. Performance Metrics**

- Test execution speed (tests/second) - derived from runtime and test count
- Parallel execution efficiency - derived from sequential vs parallel runtime
- Runtime trends over time - derived from historical runtime data

**2. Health Trends**

- Pass rate (passed / total) - derived from pass/fail counts
- Failure rate - derived from pass/fail counts
- Coverage trends over time - derived from historical coverage data
- Flaky test rerun rate - derived from flaky test count and total runs

### Machine-Readable Artifacts

**Minimum Set (Always Emit in CI):**

1. **JUnit XML** (`--junitxml=reports/junit.xml`)

   - Pass/fail/skip status
   - Test timing information
   - Enables aggregation and trend analysis

2. **Coverage Reports**

   - Terminal summary (`--cov-report=term-missing`)
   - HTML report (`--cov-report=html`) for deep inspection
   - XML report (`--cov-report=xml`) for CI integration

3. **Slowest Tests** (`--durations=20`)

   - Identifies performance bottlenecks
   - Enables targeted optimization

**Recommended CI Command:**

```bash
pytest -m "unit or integration" \
  --durations=20 \
  --junitxml=reports/junit.xml \
  --cov=podcast_scraper \
  --cov-report=xml \
  --cov-report=term \
  --cov-report=html
```text

  --json-report --json-report-file=reports/pytest.json \
  --durations=20 \
  --junitxml=reports/junit.xml \
  --cov=podcast_scraper --cov-report=xml

```text
- Enables automated trend analysis

## GitHub Actions Integration

### Pull Requests

**Tier 1 Tests (Fast Confidence):**

- Run unit + integration tests
- Upload JUnit XML + coverage HTML as artifacts
- Print slowest 20 tests in logs
- Generate GitHub Actions job summary

**Job Summary Example:**

- Total tests: 250
- Passed: 248, Failed: 0, Skipped: 2
- Total runtime: 2m 15s
- Coverage: 65.3%
- Slowest test: `test_full_pipeline` (12.3s)

### Main Branch (Layer 2)

**Tier 1 + Tier 2 Tests (Full Validation):**

- Run all tests (unit + integration + E2E)
- Track flaky tests (reruns enabled)
- Generate basic metrics (JUnit XML, coverage reports)
- Upload artifacts for download
- Job summaries with key metrics

**Current Implementation:**
- `lint` job: Fast checks (1-2 min)
- `test-unit` job: Unit tests only (2-5 min)
- `test-integration` job: All integration tests with re-runs (5-10 min)
- `test-e2e` job: All E2E tests with re-runs and network guard (20-30 min)
- `docs` job: Documentation build (2-3 min)
- `build` job: Package build (1-2 min)
- All jobs run in parallel

### Nightly / Scheduled (Layer 3)

**Full Suite + Comprehensive Analysis:**

- **Complete test suite**: Everything that main branch does (lint, test-unit, test-integration, test-e2e, docs, build)
- **Comprehensive metrics collection**:
  - JUnit XML for all test tiers
  - Coverage XML/HTML reports
  - Slowest tests identification (`--durations=20`)
  - pytest-json-report for structured metrics
- **Trend tracking**:
  - Append metrics to history file (CSV/JSONL)
  - Store on dedicated branch or gh-pages
  - Enable trend visualization
- **Reporting**:
  - Generate comprehensive job summaries
  - Create metrics dashboards
  - Performance regression detection
  - Flaky test analysis and reporting
- **Additional requirements**:
  - Full artifact preservation
  - Historical data aggregation
  - Automated trend analysis
  - Regression alerts (optional)

**Implementation Plan:**
- Scheduled workflow (nightly at 2 AM UTC)
- Runs all main branch jobs plus metrics collection
- Generates comprehensive reports
- Stores metrics for trend tracking

### GitHub Actions Job Summary

Use `$GITHUB_STEP_SUMMARY` to create automatic dashboards:

```bash

echo "## Test Results" >> $GITHUB_STEP_SUMMARY
echo "- Total: $(jq '.summary.total' reports/pytest.json)" >> $GITHUB_STEP_SUMMARY
echo "- Passed: $(jq '.summary.passed' reports/pytest.json)" >> $GITHUB_STEP_SUMMARY
echo "- Failed: $(jq '.summary.failed' reports/pytest.json)" >> $GITHUB_STEP_SUMMARY
echo "- Runtime: $(jq '.duration' reports/pytest.json)s" >> $GITHUB_STEP_SUMMARY
echo "- Coverage: $(coverage report --format=total)" >> $GITHUB_STEP_SUMMARY

```text
- No additional infrastructure required

**Pros:**

- Zero maintenance
- No external dependencies
- Historical data preserved

**Cons:**

- Manual comparison required
- No automatic trend visualization

### Option B: Lightweight History File (Recommended)

**Implementation:**

- Append one row per run to `metrics/history.csv` (or JSONL)
- Store on `gh-pages` branch or dedicated metrics branch
- Columns: `date`, `commit`, `tier_runtime`, `passed`, `failed`, `skipped`, `coverage`, `flaky_count`

**Example CSV:**

```csv

date,commit,tier0_runtime,tier1_runtime,tier2_runtime,passed,failed,skipped,coverage,flaky_count
2024-12-28T19:00:00Z,abc123,2.1,33.6,0,248,0,2,65.3,0
2024-12-28T20:00:00Z,def456,2.2,34.1,0,249,0,2,65.5,0

```text
- Minimal maintenance
- Version-controlled history

### Option C: External Metrics Service (Future)

**Future Enhancement:**

- Integrate with metrics service (e.g., Datadog, Grafana)
- Automatic dashboards and alerts
- Advanced analytics and regression detection

**When to Consider:**

- When project scales significantly
- When multiple contributors need metrics access
- When automated alerts are needed

## Flaky Test Detection

### Definition

**Flaky Test:** A test is flaky if it:
- Fails and passes on rerun without code changes, OR
- Has < 95% pass rate over last N runs

This explicit definition enables automation and clear identification of unstable tests.

### Detection Methods

**Method 1: Rerun-on-Failure (Current)**

- Use `pytest-rerunfailures` with `--reruns 2 --reruns-delay 1`
- Tests that pass on rerun are counted as flaky
- Track flaky count in metrics

**Method 2: Quarantine Marker**

- Mark known flaky tests with `@pytest.mark.flaky`
- Track number of quarantined tests
- Monitor failure rate of quarantined tests over time

**Method 3: Historical Analysis**

- Track test pass/fail history over multiple runs
- Identify tests with inconsistent results
- Flag tests with < 95% pass rate over last N runs as flaky (matches definition)

### Reporting

**Include in Metrics:**

- Total flaky test count
- Flaky test names
- Flaky test failure rate
- Trend over time (improving or degrading)

## CI Layer Strategy

### Layer 1: Pull Requests (Fast Feedback)

**Purpose:** Quick validation for PRs

**What runs:**
- Fast tests only (unit + fast integration + fast e2e)
- Basic metrics (JUnit XML, coverage)
- Job summaries

**Current Status:** ✅ Implemented

### Layer 2: Main Branch (Full Validation)

**Purpose:** Complete validation on merge to main

**What runs:**
- All tests (unit + integration + e2e, including slow/ml_models)
- Basic metrics (JUnit XML, coverage)
- Artifact uploads
- Job summaries

**Current Status:** ✅ Implemented

### Layer 3: Nightly Builds (Comprehensive Analysis)

**Purpose:** Comprehensive metrics, reporting, and trend tracking

**What runs:**
- **Everything from Layer 2** (lint, test-unit, test-integration, test-e2e, docs, build)
- **Comprehensive metrics collection**:
  - JUnit XML for all test tiers
  - Coverage XML/HTML reports
  - Slowest tests (`--durations=20`)
  - pytest-json-report for structured metrics
- **Trend tracking**:
  - Metrics history file (CSV/JSONL)
  - Historical data aggregation
  - Trend visualization
- **Reporting**:
  - Comprehensive job summaries
  - Metrics dashboards
  - Performance regression detection
  - Flaky test analysis
- **Additional requirements**:
  - Full artifact preservation
  - Automated trend analysis
  - Regression alerts (optional)

**Current Status:** 🚧 **To Be Implemented**

## Implementation Plan

### Phase 1: Basic Metrics Collection (Layer 2 Enhancement)

**Goal:** Add basic metrics to main branch runs

- [ ] Ensure JUnit XML generation in CI
- [ ] Ensure coverage XML/HTML generation in CI
- [ ] Add `--durations=20` to CI test commands
- [ ] Upload test artifacts in GitHub Actions

**Estimated Time:** 1-2 days

**Status:** 🚧 In Progress

### Phase 2: GitHub Actions Job Summary (Layer 2 Enhancement)

**Goal:** Display key metrics in job summaries

- [ ] Create job summary script
- [ ] Extract metrics from JUnit XML and coverage reports
- [ ] Format and display in GitHub Actions summary
- [ ] Test on PR and main branch runs

**Estimated Time:** 1-2 days

**Status:** 🚧 To Be Implemented

### Phase 3: Layer 3 - Nightly Builds (Comprehensive Analysis)

**Goal:** Implement comprehensive nightly builds with full metrics and reporting

**Tasks:**

- [ ] **Create nightly workflow**
  - Scheduled trigger (nightly at 2 AM UTC)
  - Runs all Layer 2 jobs plus metrics collection
  - Comprehensive reporting

- [ ] **Enhanced metrics collection**
  - Add `pytest-json-report` for structured metrics
  - Generate comprehensive JUnit XML for all tiers
  - Generate coverage XML/HTML for all tiers
  - Collect slowest tests from all tiers

- [ ] **Trend tracking implementation**
  - Create metrics history file (CSV/JSONL)
  - Script to append metrics per run
  - Store on dedicated branch or gh-pages
  - Enable trend visualization

- [ ] **Comprehensive reporting**
  - Generate detailed job summaries
  - Create metrics dashboards
  - Performance regression detection
  - Flaky test analysis and reporting

- [ ] **Additional requirements**
  - Full artifact preservation (extended retention)
  - Historical data aggregation
  - Automated trend analysis
  - Regression alerts (optional)

**Estimated Time:** 3-5 days

**Status:** 🚧 To Be Implemented

### Phase 4: Enhanced Metrics (Layer 3 Enhancement)

**Goal:** Advanced metrics and analysis

- [ ] Implement flaky test detection and reporting
- [ ] Create automated trend analysis
- [ ] Set up alerts for regressions (optional)
- [ ] Advanced visualization dashboards

**Estimated Time:** 2-3 days

**Status:** 🚧 Future Enhancement

## Design Decisions

### 1. Metrics Collection Strategy

**Decision:** Start with artifacts, add history file later

**Rationale:**

- Artifacts: Zero maintenance, immediate value
- History file: Enables trends without external services
- Future: Can add external service if needed

### 2. Flaky Test Detection

**Decision:** Use rerun-on-failure + quarantine markers

**Rationale:**

- Rerun-on-failure: Automatic detection
- Quarantine markers: Explicit acknowledgment
- Historical analysis: Future enhancement

## Benefits

### Codebase Health

- ✅ **Visibility**: Key metrics tracked on every run
- ✅ **Trends**: Historical data enables improvement tracking
- ✅ **Regression detection**: Identify slow tests and performance issues
- ✅ **Data-driven decisions**: Metrics guide optimization efforts

### CI/CD Integration

- ✅ **Automated metrics**: No manual work required
- ✅ **Artifact preservation**: Historical data available
- ✅ **Job summaries**: Always-visible health snapshots
- ✅ **Trend analysis**: Track progress over time

## Related Files

- `.github/workflows/python-app.yml`: CI test jobs
- `docs/TESTING_STRATEGY.md`: Overall testing strategy
- `docs/guides/DEVELOPMENT_GUIDE.md`: Development workflow
- `pyproject.toml`: Pytest configuration and markers

## Current Implementation Status

### ✅ Layer 1 (PRs) - Implemented

- Fast feedback jobs (`test-fast` and `test`)
- Basic test execution
- Parallel execution for speed

### ✅ Layer 2 (Main Branch) - Implemented

- All test jobs (test-unit, test-integration, test-e2e)
- Complete validation
- Parallel execution for speed

### 🚧 Layer 3 (Nightly Builds) - To Be Implemented

- Comprehensive metrics collection
- Trend tracking
- Reporting and dashboards
- Performance regression detection

## Notes

- Metrics collection should be low-friction (automated, no manual work)
- Trend tracking can start simple (artifacts) and evolve (history file, external service)
- Flaky test detection improves over time with historical data
- Layer 3 (nightly builds) provides comprehensive analysis without slowing down PR or main branch CI
- See `RFC-024-test-execution-optimization.md` for test execution optimization strategy
````
