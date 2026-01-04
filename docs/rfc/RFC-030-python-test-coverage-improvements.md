# RFC-030: Python Test Coverage Improvements

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, developers, CI/CD pipeline maintainers
- **Related PRDs**:
  - `docs/prd/PRD-001-transcript-pipeline.md` (core pipeline)
- **Related RFCs**:
  - `docs/rfc/RFC-025-test-metrics-and-health-tracking.md` (metrics collection)
  - `docs/rfc/RFC-026-metrics-consumption-and-dashboards.md` (metrics consumption)
  - `docs/rfc/RFC-024-test-execution-optimization.md` (test execution optimization)
  - `docs/rfc/RFC-018-test-structure-reorganization.md` (test structure)
- **Related Documents**:
  - `docs/TESTING_STRATEGY.md` - Overall testing strategy
  - `docs/guides/DEVELOPMENT_GUIDE.md` - Development workflow
  - `.github/workflows/python-app.yml` - Main CI workflow
  - `.github/workflows/nightly.yml` - Nightly comprehensive tests

## Abstract

This RFC proposes improvements to Python test coverage collection, reporting, and enforcement.
Currently, coverage is only comprehensively collected in nightly builds. This RFC addresses gaps
in PR-level feedback, coverage threshold enforcement, and CI integration to provide developers
with immediate coverage visibility and prevent coverage regression.

**Key Improvements:**

1. Add coverage collection to regular CI workflows (not just nightly)
2. Enforce minimum coverage threshold to prevent regression
3. Provide coverage feedback in PR job summaries
4. Enable nightly schedule for historical tracking
5. Create unified coverage reports across test tiers

## Problem Statement

### Current State

The project has basic coverage infrastructure in place:

- `pyproject.toml` configures coverage with branch coverage enabled
- Makefile targets include `--cov` flags
- Nightly workflow generates comprehensive coverage reports
- Scripts exist for metrics generation and dashboard creation

### Gaps Identified

**1. No Coverage Collection in Regular CI (HIGH IMPACT)**

The main `python-app.yml` workflow runs tests but **does not collect coverage**:

```yaml

# Current: Tests run WITHOUT coverage flags

OUTPUT=$(pytest tests/unit/ -v --tb=short -n ... 2>&1)
```

Impact: No visibility into coverage changes during development or PR review.

**2. No Coverage Threshold Enforcement (MEDIUM IMPACT)**

No `fail_under` threshold configured. Coverage can regress without CI failure:

```toml
[tool.coverage.report]
show_missing = true
skip_covered = true
precision = 2

# Missing: fail_under = 65

```

Impact: Coverage erosion over time without detection.

**3. No Coverage Feedback in PRs (HIGH IMPACT)**

PR authors don't see coverage impact of their changes. RFC-026 Phase 0 goal is not implemented:

- No GitHub Job Summary with coverage
- No coverage diff or trend information
- Developers must wait for nightly or run locally

Impact: Reduced developer awareness of test coverage.

**4. Nightly Schedule Disabled (MEDIUM IMPACT)**

The nightly workflow is set to manual trigger only:

```yaml
on:
  # schedule:
  #   - cron: '0 2 * * *'  # DISABLED
  workflow_dispatch:
```

Impact: No automatic historical metrics collection.

**5. No Unified Coverage Report (MEDIUM IMPACT)**

Unit, integration, and E2E tests run in separate CI jobs without coverage combination:

- Each job generates partial coverage
- No merged view of total coverage
- Module coverage may appear lower than actual

Impact: Incomplete picture of total test coverage.

**6. No Coverage Artifacts in Regular CI (LOW IMPACT)**

Only nightly uploads coverage artifacts. Regular CI doesn't preserve coverage for debugging.

Impact: Harder to debug coverage issues in PR builds.

## Goals

### Primary Goals

1. **Immediate coverage feedback**: Developers see coverage impact in every PR
2. **Prevent coverage regression**: CI fails if coverage drops below threshold
3. **Historical tracking**: Automatic nightly collection of coverage trends
4. **Unified reporting**: Single coverage number across all test tiers

### Success Criteria

- ✅ Coverage percentage visible in GitHub Job Summary for every PR
- ✅ CI fails if coverage drops below configured threshold
- ✅ Nightly builds run automatically and collect historical data
- ✅ Combined coverage report available across test tiers
- ✅ Coverage artifacts uploaded for debugging

## Solution Design

### Phase 1: Coverage Threshold Enforcement (Quick Win)

**Effort:** 15 minutes

Add `fail_under` to `pyproject.toml`:

```toml
[tool.coverage.report]
show_missing = true
skip_covered = true
precision = 2
fail_under = 65  # Fail if coverage drops below 65%

# Optional: Exclude test files and scripts from coverage

[tool.coverage.run]
branch = true
source = ["podcast_scraper"]
omit = [
    "*/tests/*",
    "scripts/*",
]
```

**Rationale:** Sets a baseline to prevent regression. Can be adjusted as coverage improves.

## Phase 2: Add Coverage to PR Workflow (High Impact)

**Effort:** 1-2 hours

Modify `test-unit` job in `.github/workflows/python-app.yml`:

```yaml
- name: Run unit tests with coverage
  run: |

    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    mkdir -p reports
    pytest tests/unit/ -v --tb=short \
      -n $(python3 -c "import os; print(max(1, (os.cpu_count() or 2) - 2))") \
      --cov=podcast_scraper \
      --cov-report=xml:reports/coverage-unit.xml \
      --cov-report=term-missing \
      --disable-socket --allow-hosts=127.0.0.1,localhost

- name: Generate coverage summary
  if: always()

  run: |
    echo "## 📊 Unit Test Coverage" >> $GITHUB_STEP_SUMMARY
    if [ -f reports/coverage-unit.xml ]; then
      COVERAGE=$(python -c "import xml.etree.ElementTree as ET; \
        tree = ET.parse('reports/coverage-unit.xml'); \
        root = tree.getroot(); \
        print(f\"{float(root.attrib.get('line-rate', 0)) * 100:.1f}%\")" 2>/dev/null || echo "N/A")
      echo "- **Coverage**: $COVERAGE" >> $GITHUB_STEP_SUMMARY
    else
      echo "- **Coverage**: Not available" >> $GITHUB_STEP_SUMMARY
    fi
```

**Benefits:**

- Immediate coverage visibility in every PR
- No additional dependencies required
- Uses existing GitHub Actions features

### Phase 3: Enable Nightly Schedule

**Effort:** 5 minutes

Uncomment the schedule in `.github/workflows/nightly.yml`:

```yaml
on:
  schedule:

    - cron: '0 2 * * *'  # Run nightly at 2 AM UTC
  workflow_dispatch:  # Keep manual trigger option

```

**Benefits:**

- Automatic historical data collection
- Trend tracking enabled
- Dashboard populated with data

### Phase 4: Unified Coverage Report (Main Branch)

**Effort:** 1-2 hours

For main branch pushes, combine coverage from all test tiers:

**Option A: Sequential with --cov-append**

```yaml
- name: Run all tests with combined coverage
  run: |

    # Run unit tests (creates .coverage)
    pytest tests/unit/ --cov=podcast_scraper --cov-report=

    # Run integration tests (appends to .coverage)
    pytest tests/integration/ --cov=podcast_scraper --cov-append --cov-report=

    # Run E2E tests (appends to .coverage)
    pytest tests/e2e/ --cov=podcast_scraper --cov-append --cov-report=

    # Generate final report
    coverage xml -o reports/coverage-combined.xml
    coverage report --fail-under=65
```

**Option B: Parallel with coverage combine**

```yaml
- name: Combine coverage from parallel jobs
  run: |

    # Download coverage artifacts from each job
    # Merge them
    coverage combine coverage-unit/.coverage coverage-integration/.coverage coverage-e2e/.coverage
    coverage xml -o reports/coverage-combined.xml
    coverage report
```

**Recommendation:** Start with Option A for simplicity.

### Phase 5: Coverage Artifacts Upload (Optional)

**Effort:** 15 minutes

Add artifact upload to regular CI jobs:

```yaml
- name: Upload coverage artifact
  if: always()

  uses: actions/upload-artifact@v4
  with:
    name: coverage-${{ github.job }}-${{ github.run_number }}
    path: |
      reports/coverage*.xml
      .coverage
    retention-days: 14
```

### Phase 6: Third-Party Integration (Future)

**Effort:** 30 minutes

Integrate with Codecov for enhanced features:

```yaml
- name: Upload to Codecov
  uses: codecov/codecov-action@v4

  with:
    files: reports/coverage.xml
    fail_ci_if_error: false
    verbose: true
```

**Benefits:**

- Coverage badges for README
- PR comments with coverage diff
- Historical trend visualization
- Branch comparison

**Consideration:** Requires Codecov account (free for open source).

## Implementation Plan

### Phase 1: Quick Wins (Day 1)

| Task | Effort | Impact |
| ------ | -------- | -------- |
| Add `fail_under = 65` to `pyproject.toml` | 15 min | Prevents regression |
| Enable nightly schedule in `nightly.yml` | 5 min | Historical tracking |

### Phase 2: PR Feedback (Day 2)

| Task | Effort | Impact |
| ------ | -------- | -------- |
| Add `--cov` flags to `test-unit` job | 30 min | Unit coverage visible |
| Add GitHub Job Summary for coverage | 30 min | Immediate feedback |
| Test on a PR | 15 min | Verify working |

### Phase 3: Enhanced Coverage (Day 3)

| Task | Effort | Impact |
| ------ | -------- | -------- |
| Add coverage to integration/E2E jobs | 1 hour | Full visibility |
| Implement unified coverage report | 1 hour | Single metric |
| Add artifact upload | 15 min | Debug support |

### Phase 4: Optional Enhancements (Future)

| Task | Effort | Impact |
| ------ | -------- | -------- |
| Integrate Codecov | 30 min | PR comments, badges |
| Add coverage badges to README | 15 min | Visibility |
| Module-level coverage thresholds | 1 hour | Granular control |

## Configuration Reference

### Recommended `pyproject.toml` Changes

```toml
[tool.coverage.run]
branch = true
source = ["podcast_scraper"]
omit = [
    "*/tests/*",
    "scripts/*",
    "*/__pycache__/*",
]

# For parallel coverage collection

parallel = true
data_file = ".coverage"

[tool.coverage.report]
show_missing = true
skip_covered = true
precision = 2
fail_under = 65
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]

[tool.coverage.xml]
output = "reports/coverage.xml"

[tool.coverage.html]
directory = "reports/coverage-html"
```

## Makefile Updates

Add a dedicated coverage target:

```makefile
coverage-report:
	# Generate combined coverage report
	pytest tests/ --cov=$(PACKAGE) --cov-report=xml:reports/coverage.xml --cov-report=html:reports/coverage-html --cov-report=term-missing
	@echo "Coverage report generated: reports/coverage.xml"
	@echo "HTML report: reports/coverage-html/index.html"

coverage-check:
	# Check coverage threshold (useful for local development)
	coverage report --fail-under=65
```yaml

## Metrics and Monitoring

### Coverage Metrics to Track

| Metric | Source | Purpose |
| -------- | -------- | --------- |
| Overall line coverage | coverage.xml | Primary health indicator |
| Branch coverage | coverage.xml | Code path completeness |
| Coverage by module | coverage.xml | Identify weak areas |
| Coverage trend | history.jsonl | Track improvement/regression |
| Uncovered lines | term-missing | Actionable improvement targets |

### Alerting Thresholds

| Level | Threshold | Action |
| ------- | ----------- | -------- |
| 🟢 Good | ≥ 80% | No action needed |
| 🟡 Warning | 65-80% | Monitor, improve gradually |
| 🔴 Failure | < 65% | CI fails, must address |

## Risks and Mitigations

### Risk 1: Coverage Overhead Slows CI

**Mitigation:** Coverage adds ~10-15% overhead. Monitor CI times and adjust:

- Run coverage only on main branch (not PRs) if too slow
- Use `--cov-report=` (no output) during collection, generate report separately
- Exclude slow E2E tests from coverage in PRs

### Risk 2: False Coverage (Tests Pass but Don't Assert)

**Mitigation:** Coverage measures execution, not quality. Complement with:

- Code review for assertion quality
- Mutation testing (future enhancement)
- Integration/E2E tests for behavior validation

### Risk 3: Threshold Too High/Low

**Mitigation:** Start with 65% (current approximate level), adjust based on:

- Actual coverage after measurement
- Team feedback
- Module-specific requirements

## Benefits

### Developer Experience

- ✅ Immediate visibility into coverage impact
- ✅ Clear threshold for minimum coverage
- ✅ Historical trends show improvement over time
- ✅ No manual coverage checks needed

### Code Quality

- ✅ Prevents coverage regression
- ✅ Identifies untested code paths
- ✅ Encourages test-first development
- ✅ Branch coverage ensures path completeness

### CI/CD Integration

- ✅ Automated collection and reporting
- ✅ GitHub Job Summaries for quick review
- ✅ Artifacts preserved for debugging
- ✅ Trend tracking via nightly builds

## Related Files

- `pyproject.toml` - Coverage configuration
- `.github/workflows/python-app.yml` - Main CI workflow
- `.github/workflows/nightly.yml` - Nightly comprehensive tests
- `scripts/generate_metrics.py` - Metrics extraction
- `scripts/generate_dashboard.py` - Dashboard generation
- `Makefile` - Development commands

## Notes

- Start with Phase 1-2 for immediate impact
- Phase 3-4 can be implemented incrementally
- Codecov integration is optional but provides nice PR experience
- Coverage threshold can be adjusted as codebase improves
- Consider module-specific thresholds for critical paths (workflow, config, etc.)
