# RFC-026: Metrics Consumption and Dashboards

- **Status**: ✅ **Completed** (Phases 0-3 complete, Phase 4 extracted to RFC-040)
- **Authors**:
- **Stakeholders**: Maintainers, developers, CI/CD pipeline maintainers
- **Completed**: 2026-01-07
- **Related PRDs**:
  - `docs/prd/PRD-001-transcript-pipeline.md` (core pipeline)
- **Related RFCs**:
  - `docs/rfc/RFC-025-test-metrics-and-health-tracking.md` (metrics collection - prerequisite)
  - `docs/rfc/RFC-024-test-execution-optimization.md` (test execution optimization)
  - **`docs/rfc/RFC-040-automated-metrics-alerts.md`** (Phase 4 - extracted for independent evolution)
- **Related Documents**:

**🚨 DEPENDENCY NOTE:**

**RFC-026 assumes RFC-024 and RFC-025 are implemented.**

This RFC builds on the test execution optimization (RFC-024) and metrics collection (RFC-025) foundations. Ensure those RFCs are implemented before proceeding with metrics consumption and dashboards.

- `docs/TESTING_STRATEGY.md` - Overall testing strategy and test categories
- `docs/guides/DEVELOPMENT_GUIDE.md` - Development workflow and testing requirements
- `.github/workflows/python-app.yml` - CI test jobs

## Abstract

This RFC defines a strategy for **consuming and visualizing test metrics** to enable quick deviation detection and trend analysis. The strategy focuses on:

1. **Easy access**: Multiple consumption methods (browser, API, PR checks)
2. **Quick detection**: Identify deviations in **< 60 seconds**
3. **Visual dashboards**: Human-readable charts and alerts
4. **Machine-readable API**: JSON endpoints for automation
5. **Zero infrastructure**: Uses GitHub Pages (free, no setup)

**Key Principle:** Metrics are only valuable if they can be consumed quickly. Enable < 60 second deviation detection through multiple access patterns.

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
- RFC-025: Metrics collection (artifacts generation)

## Core Principles

These principles are shared across RFC-024, RFC-025, and RFC-026:

- **Developer flow > completeness** - Fast feedback loops protect developer state and enable rapid iteration
- **Metrics must be cheap to collect** - Automated collection with zero manual work required
- **Humans consume summaries, machines consume JSON** - Job summaries for quick checks, JSON API for automation

## Problem Statement

**Current Issues:**

1. **No Easy Metrics Access**
   - Metrics exist in CI artifacts but require manual download
   - No public dashboard for quick checks
   - No machine-readable API for automation
   - Historical trends not easily accessible

2. **Slow Deviation Detection**
   - Manual comparison of artifacts takes minutes
   - No automatic alerts for regressions
   - Difficult to spot trends without visualization
   - No quick way to check if metrics are degrading

**Impact:**

- Developers don't check metrics regularly (too much effort)
- Regressions go undetected until they become severe
- No visibility into long-term trends
- Difficult to make data-driven optimization decisions

## Goals

### Primary Goal

**Quick Metrics Consumption:**

- Enable deviation detection in **< 60 seconds**
- Multiple access methods (browser, API, PR checks)
- Automatic alerts for regressions
- Visual dashboards for trend analysis
- Zero infrastructure overhead (GitHub Pages)

### Success Criteria

- ✅ Metrics accessible via public URL (< 10 seconds to view)
- ✅ JSON API for automation (< 5 seconds to query)
- ✅ Deviation detection in < 60 seconds
- ✅ Visual dashboards with trend charts
- ✅ Automatic alerts for regressions
- ✅ Historical data available for analysis

## Solution: GitHub Pages Unified Metrics Dashboard

**Approach:** Publish metrics to GitHub Pages as both human-readable unified dashboard and machine-readable JSON.

**Benefits:**

- ✅ **Always accessible**: Public URL (e.g., `https://chipi.github.io/podcast_scraper/metrics/`)
- ✅ **No authentication**: Anyone can view metrics
- ✅ **Auto-updated**: Metrics published after each CI run and nightly schedule
- ✅ **Unified interface**: Single dashboard with data source selector (CI or Nightly)
- ✅ **Quick consumption**: View dashboard in browser (< 10 seconds)
- ✅ **Machine-readable**: JSON API for automation (separate files for CI and Nightly)
- ✅ **Historical trends**: Visual charts showing deviations (last 30 runs per source)
- ✅ **Zero infrastructure**: Uses GitHub Pages (free, no setup)

**Dashboard Features:**

- **Data Source Selector**: Dropdown to switch between CI Metrics and Nightly Metrics
- **Auto-detection**: Automatically loads available data source on page load
- **Dynamic Loading**: JavaScript fetches appropriate JSON files based on selection
- **Same Features**: All dashboard features work for both data sources (charts, alerts, slowest tests, etc.)

## Implementation Strategy

### Phase 0: Minimum Viable Consumption (Mandatory, Before Dashboards)

**🚨 CRITICAL: This phase must be completed before any dashboard work.**

**Goal:** Enable basic metrics consumption without visual dashboards.

**Deliverables:**

- ✅ **GitHub Actions job summaries** - Display key metrics in PR checks (0 seconds to view)
- ✅ **`metrics/latest.json` published** - Machine-readable metrics available via GitHub Pages
- ❌ **No charts** - Visual dashboards are not required in this phase
- ❌ **No history UI** - Historical visualization is not required in this phase

**Rationale:**

- **Summaries ≫ dashboards** - Job summaries provide immediate value with zero infrastructure
- **Dashboards are earned, not required** - Visual dashboards come after basic consumption is proven
- **Focus on consumption, not visualization** - Enable metrics access first, add visuals later

**Success Criteria:**

- ✅ Job summaries show key metrics (runtime, coverage, pass rate) in every PR
- ✅ `metrics/latest-ci.json` and `metrics/latest-nightly.json` are accessible via public URL
- ✅ Metrics can be consumed via `curl` + `jq` in < 5 seconds
- ✅ No visual dashboard required

**Status:** 🚧 To Be Implemented (prerequisite for all other phases)

### 1. Metrics JSON API (Machine-Readable)

**Locations:**
- CI Metrics: `https://chipi.github.io/podcast_scraper/metrics/latest-ci.json`
- Nightly Metrics: `https://chipi.github.io/podcast_scraper/metrics/latest-nightly.json`
- CI History: `https://chipi.github.io/podcast_scraper/metrics/history-ci.jsonl`
- Nightly History: `https://chipi.github.io/podcast_scraper/metrics/history-nightly.jsonl`

**Format:**

```json
{
  "timestamp": "2024-12-28T20:00:00Z",
  "commit": "def456",
  "branch": "main",
  "workflow_run": "https://github.com/chipi/podcast_scraper/actions/runs/12345",
  "metrics": {
    "runtime": {
      "unit_tests": 2.1,
      "integration_tests": 33.6,
      "e2e_tests": 0,
      "total": 35.7
    },
    "test_health": {
      "total": 250,
      "passed": 248,
      "failed": 0,
      "skipped": 2,
      "flaky": 0,
      "pass_rate": 0.992
    },
    "coverage": {
      "overall": 65.3,
      "by_module": {
        "podcast_scraper": 65.3,
        "podcast_scraper.workflow": 72.1
      }
    },
    "performance": {
      "tests_per_second": 7.0,
      "parallel_efficiency": 0.95
    },
    "slowest_tests": [
      {"name": "test_full_pipeline", "duration": 12.3},
      {"name": "test_transcription", "duration": 8.7}
    ]
  },
  "trends": {
    "runtime_change": "+0.5s",
    "coverage_change": "+0.2%",
    "test_count_change": "+1"
  },
  "alerts": [
    {
      "type": "regression",
      "metric": "runtime",
      "severity": "warning",
      "message": "Runtime increased by 15% compared to last 5 runs"
    }
  ]
}
```bash

# Fetch CI metrics

curl -s https://chipi.github.io/podcast_scraper/metrics/latest-ci.json | jq '.metrics.runtime.total'

# Fetch nightly metrics

curl -s https://chipi.github.io/podcast_scraper/metrics/latest-nightly.json | jq '.metrics.runtime.total'

# Check for regressions (CI)

curl -s https://chipi.github.io/podcast_scraper/metrics/latest-ci.json | jq '.alerts[]'

# Check for regressions (Nightly)

curl -s https://chipi.github.io/podcast_scraper/metrics/latest-nightly.json | jq '.alerts[]'

```text

- **Current metrics** (latest run)
- **Trend charts** (last 30 runs)
- **Deviation alerts** (highlighted in red/yellow)
- **Quick comparison** (vs. previous run, vs. baseline)
- **Slowest tests** (top 10)
- **Coverage trends** (visual chart)

**Visual Elements:**

- ✅ Green: Metrics within normal range
- ⚠️ Yellow: Minor deviation (< 10%)
- 🔴 Red: Significant deviation (> 10%)
- 📊 Charts: Line graphs for trends

**Example Dashboard:**

```html

<!-- Simplified example -->
<div class="metrics-dashboard">
  <h1>Test Metrics Dashboard</h1>

  <div class="current-metrics">
    <h2>Latest Run (2024-12-28 20:00:00)</h2>
    <div class="metric">
      <span>Runtime:</span> 35.7s <span class="trend up">+0.5s</span>
    </div>
    <div class="metric">
      <span>Coverage:</span> 65.3% <span class="trend up">+0.2%</span>
    </div>
    <div class="metric">
      <span>Tests:</span> 250 <span class="status pass">248 passed</span>
    </div>
  </div>

  <div class="trends">
    <h2>Trends (Last 30 Runs)</h2>
    <canvas id="runtime-chart"></canvas>
    <canvas id="coverage-chart"></canvas>
  </div>

  <div class="alerts">
    <h2>Alerts</h2>
    <div class="alert warning">
      Runtime increased by 15% compared to baseline
    </div>
  </div>
</div>

```json

{"timestamp":"2024-12-28T19:00:00Z","commit":"abc123","runtime":35.2,"coverage":65.1,"passed":248}
{"timestamp":"2024-12-28T20:00:00Z","commit":"def456","runtime":35.7,"coverage":65.3,"passed":248}

```text

- Can append without rewriting entire file

## 4. GitHub Actions Integration

**Workflow Step:**

```yaml

- name: Generate and publish metrics
  if: always() && github.ref == 'refs/heads/main'

  run: |

    # Extract metrics from JUnit XML and coverage

    python scripts/generate_metrics.py \
      --junit reports/junit.xml \
      --coverage reports/coverage.xml \
      --output metrics/

    # Generate HTML dashboard

    python scripts/generate_dashboard.py \
      --metrics metrics/latest.json \
      --history metrics/history.jsonl \
      --output metrics/index.html

```text

    # Publish to gh-pages branch

```bash

    git checkout gh-pages || git checkout --orphan gh-pages
    git add metrics/
    git commit -m "Update metrics: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git push origin gh-pages

```text

3. Check trend charts for spikes

### Method 2: JSON API (5 seconds)

```bash

# Check latest metrics

curl -s https://chipi.github.io/podcast_scraper/metrics/latest.json | jq '.alerts'

# Compare with previous run

curl -s https://chipi.github.io/podcast_scraper/metrics/latest.json | jq '.trends'

```text

- No external access needed

## Method 4: Automated Alerts (0 seconds)

- GitHub Actions can comment on PRs with metric changes
- Slack/Discord webhooks for significant deviations
- Email notifications (optional)

## Deviation Detection Logic

### Thresholds

- **Minor deviation**: 5-10% change
- **Significant deviation**: > 10% change
- **Critical deviation**: > 20% change

### Alert Behavior

**🚨 CRITICAL: Alerts are informational initially (no CI failures)**

- Alerts are displayed in job summaries and dashboards
- Alerts do NOT cause CI failures or block merges
- Alerts are informational only - they highlight potential issues for review
- This prevents teams from fearing noise and disabling alerts

**Future Enhancement:** After alerts are proven useful and accurate, consider optional CI gates (opt-in per team).

### Metrics to Monitor

1. **Runtime**: Compare against last 5 runs (median)
2. **Coverage**: Compare against last 10 runs (trend)
3. **Test count**: Alert if tests added/removed
4. **Slowest tests**: Alert if new slow tests appear
5. **Flaky tests**: Alert if flaky count increases

### Example Detection

```python

def detect_deviations(current, history):
    alerts = []

    # Runtime deviation

    median_runtime = median([r['runtime'] for r in history[-5:]])
    if current['runtime'] > median_runtime * 1.1:
        alerts.append({
            "type": "regression",
            "metric": "runtime",
            "severity": "warning",
            "message": f"Runtime increased by {((current['runtime'] / median_runtime) - 1) * 100:.1f}%"
        })

    # Coverage drop

    avg_coverage = mean([r['coverage'] for r in history[-10:]])
    if current['coverage'] < avg_coverage - 1.0:
        alerts.append({
            "type": "regression",
            "metric": "coverage",
            "severity": "error",
            "message": f"Coverage dropped by {avg_coverage - current['coverage']:.1f}%"
        })

```text

    return alerts

```python

- Simple deviation detection

**Deliverables:**

- `scripts/generate_metrics.py` - Extract metrics from JUnit/coverage
- GitHub Actions step to publish to gh-pages
- `metrics/latest-ci.json` and `metrics/latest-nightly.json` accessible via GitHub Pages

### Phase 2: HTML Dashboard (2-3 days)

- Generate HTML dashboard
- Add trend charts (using Chart.js or similar)
- Visual alerts and highlights

**Deliverables:**

- `scripts/generate_dashboard.py` - Generate HTML dashboard
- `metrics/index.html` with charts and alerts
- CSS styling for visual indicators

### Phase 3: Historical Tracking (1-2 days)

- Append to `history.jsonl` on each run
- Load history into dashboard
- Show trend lines

**Deliverables:**

- Append logic to `generate_metrics.py`
- Dashboard loads and displays historical data
- Trend charts show last 30 runs

### Phase 4: Automated Alerts (MOVED TO RFC-040)

**Status:** ⏭️ **Extracted to RFC-040** for independent evolution

Phase 4 (automated alerts) has been extracted to a separate RFC to enable:
- Independent evolution of alerting strategy
- Clear completion milestone for RFC-026 (Phases 0-3)
- Focused implementation tracking in v2.7 milestone

**See:** [RFC-040: Automated Metrics Alerts](RFC-040-automated-metrics-alerts.md)

**Original scope (now in RFC-040):**
- PR comments on metric changes
- Webhook notifications (optional)
- Email alerts (optional)

**Rationale for extraction:**
- Phases 0-3 are complete and production-ready
- Phase 4 is substantial work (~1 day) not yet started
- Separating allows RFC-026 to be marked as complete
- Issue #216 now tracks RFC-040 implementation

## Access Patterns

### Consumption Methods by Audience

| Method | Audience | Use Case | Access Time |
| -------- | ---------- | ---------- | ------------- |
| **Job Summary** | PR authors | "Did I break something?" | 0s (view in PR checks) |
| **JSON API** | Automation | Gates, scripts, CI integration | 5s (`curl` + `jq`) |
| **Unified Dashboard** | Maintainers | Trend spotting, historical analysis, compare CI vs Nightly | 10s (browser) |

**Rationale:**

- **Job Summary** - Immediate feedback for PR authors checking if their changes broke tests
- **JSON API** - Machine-readable for automation, gates, and scripts (separate endpoints for CI and Nightly)
- **Unified Dashboard** - Visual tool for maintainers to spot trends, analyze historical data, and compare CI vs Nightly metrics using the data source selector

### For Quick Checks (< 60 seconds)

1. **GitHub Actions Job Summary** (0s) - View in PR checks
2. **JSON API** (5s) - `curl` + `jq` for automation
3. **HTML Dashboard** (10s) - Browser for visual inspection

### For Deep Analysis

- Download `history.jsonl` for custom analysis
- Use JSON API for integration with other tools
- Export to CSV for spreadsheet analysis

## Design Decisions

### 1. GitHub Pages vs. External Service

**Decision:** Use GitHub Pages for metrics publishing

**Rationale:**

- Zero infrastructure overhead
- Free and always available
- No authentication required
- Version-controlled history
- Easy to set up and maintain

**Future:** Can migrate to external service (Datadog, Grafana) if needed

### 2. JSONL vs. CSV for History

**Decision:** Use JSONL (JSON Lines) format

**Rationale:**

- Easy to append (no file rewrite)
- Machine-readable (JSON)
- Efficient for streaming
- Can parse line-by-line
- More flexible than CSV

### 3. Dashboard Technology

**Decision:** Static HTML with Chart.js (or similar)

**Rationale:**

- No server-side rendering needed
- Works with GitHub Pages (static hosting)
- Lightweight and fast
- Easy to customize
- No dependencies on external services

## Benefits

### Developer Experience

- ✅ **Quick access**: View metrics in < 10 seconds
- ✅ **Visual insights**: Charts show trends at a glance
- ✅ **Automatic alerts**: Regressions highlighted automatically
- ✅ **Multiple methods**: Browser, API, or PR checks

### Automation

- ✅ **JSON API**: Easy integration with other tools
- ✅ **Webhooks**: Can trigger alerts on deviations
- ✅ **CI integration**: Metrics published automatically
- ✅ **Historical data**: Available for custom analysis

### Maintenance

- ✅ **Zero infrastructure**: Uses GitHub Pages
- ✅ **Auto-updates**: Metrics published after each CI run
- ✅ **Version-controlled**: History stored in git
- ✅ **Low maintenance**: Minimal ongoing work

## Related Files

- `.github/workflows/python-app.yml`: CI test jobs
- `scripts/generate_metrics.py`: Extract metrics from test artifacts (to be created)
- `scripts/generate_dashboard.py`: Generate HTML dashboard (to be created)
- `docs/rfc/RFC-025-test-metrics-and-health-tracking.md`: Metrics collection (prerequisite)

## Notes

- Requires RFC-025 Phase 1 (Basic Metrics Collection) to be completed first
- GitHub Pages must be enabled for the repository
- Metrics are public (no authentication)
- Historical data grows over time (consider retention policy)

````
