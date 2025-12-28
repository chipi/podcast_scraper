# RFC-025: Test Metrics and Health Tracking

- **Status**: Draft
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
  - `docs/DEVELOPMENT_NOTES.md` - Development workflow and testing requirements
  - `.github/workflows/python-app.yml` - CI test jobs

## Abstract

This RFC defines a strategy for **tracking test metrics and monitoring codebase health** over time. The strategy focuses on:

1. **Metrics collection**: What to track on every test run
2. **CI integration**: How to collect and display metrics automatically
3. **Trend tracking**: How to monitor health over time
4. **Flaky test detection**: How to identify and track unstable tests

**Key Principle:** Metrics enable improvement. Track runtime, coverage, flakiness, and trends over time to enable data-driven optimization decisions.

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

Track these metrics on every test run to monitor codebase health:

1. **Runtime Metrics**

   - Total runtime per tier (unit / integration / e2e)
   - Top 20 slowest tests
   - Runtime trends over time

2. **Test Health Metrics**

   - Pass rate (passed / total)
   - Failure rate
   - Skipped test count
   - Flaky test count and rerun rate

3. **Coverage Metrics**

   - Overall coverage percentage
   - Coverage by module
   - Coverage trends over time
   - Uncovered lines identification

4. **Performance Metrics**

   - Test execution speed (tests/second)
   - Parallel execution efficiency
   - Resource usage (CPU, memory)

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
```

**JSON Report for Easy Aggregation:**

Add `pytest-json-report` for structured metrics:

```bash
pytest -m "unit or integration" \
  --json-report --json-report-file=reports/pytest.json \
  --durations=20 \
  --junitxml=reports/junit.xml \
  --cov=podcast_scraper --cov-report=xml
```

- Single file with all metrics
- Easy to parse and aggregate
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

### Main Branch

**Tier 1 + Tier 2 Tests (Full Validation):**

- Run all tests (unit + integration + E2E)
- Track flaky tests (reruns enabled)
- Generate comprehensive metrics
- Store metrics for trend analysis

### Nightly / Scheduled

**Full Suite + Analysis:**

- Complete test suite
- Runtime and flake trend analysis
- Coverage trend analysis
- Performance regression detection

### GitHub Actions Job Summary

Use `$GITHUB_STEP_SUMMARY` to create automatic dashboards:

```bash
echo "## Test Results" >> $GITHUB_STEP_SUMMARY
echo "- Total: $(jq '.summary.total' reports/pytest.json)" >> $GITHUB_STEP_SUMMARY
echo "- Passed: $(jq '.summary.passed' reports/pytest.json)" >> $GITHUB_STEP_SUMMARY
echo "- Failed: $(jq '.summary.failed' reports/pytest.json)" >> $GITHUB_STEP_SUMMARY
echo "- Runtime: $(jq '.duration' reports/pytest.json)s" >> $GITHUB_STEP_SUMMARY
echo "- Coverage: $(coverage report --format=total)" >> $GITHUB_STEP_SUMMARY
```

**Note:** For easy metrics consumption, dashboards, and quick deviation detection, see **RFC-026: Metrics Consumption and Dashboards** (`docs/rfc/RFC-026-metrics-consumption-and-dashboards.md`).

## Trend Tracking

### Option A: Artifacts Only (Zero Maintenance)

- Upload `reports/` directory as artifact every CI run
- Compare runs by downloading artifacts when needed
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
```

- Real trend lines without external services
- Easy to visualize (CSV → charts)
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

**Flaky Test:** A test that passes and fails non-deterministically for the same code.

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
- Flag tests with < 95% pass rate as potentially flaky

### Reporting

**Include in Metrics:**

- Total flaky test count
- Flaky test names
- Flaky test failure rate
- Trend over time (improving or degrading)

## Implementation Plan

### Phase 1: Basic Metrics Collection (Short-term)

- [ ] Ensure JUnit XML generation in CI
- [ ] Ensure coverage XML/HTML generation in CI
- [ ] Add `--durations=20` to CI test commands
- [ ] Upload test artifacts in GitHub Actions

**Estimated Time:** 1-2 days

### Phase 2: GitHub Actions Job Summary (Short-term)

- [ ] Create job summary script
- [ ] Extract metrics from JUnit XML and coverage reports
- [ ] Format and display in GitHub Actions summary
- [ ] Test on PR and main branch runs

**Estimated Time:** 1-2 days

### Phase 3: Trend Tracking (Medium-term)

- [ ] Implement metrics history file (CSV or JSONL)
- [ ] Create script to append metrics per run
- [ ] Store on dedicated branch or gh-pages
- [ ] Create simple visualization (optional)

**Estimated Time:** 2-3 days

### Phase 4: Enhanced Metrics (Optional, Long-term)

- [ ] Add `pytest-json-report` for structured metrics
- [ ] Implement flaky test detection and reporting
- [ ] Create automated trend analysis
- [ ] Set up alerts for regressions (optional)

**Estimated Time:** 3-5 days

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
- `docs/DEVELOPMENT_NOTES.md`: Development workflow
- `pyproject.toml`: Pytest configuration and markers

## Notes

- Metrics collection should be low-friction (automated, no manual work)
- Trend tracking can start simple (artifacts) and evolve (history file, external service)
- Flaky test detection improves over time with historical data
- See `RFC-024-test-execution-optimization.md` for test execution optimization strategy
