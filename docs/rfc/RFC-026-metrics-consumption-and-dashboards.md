# RFC-026: Metrics Consumption and Dashboards

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, developers, CI/CD pipeline maintainers
- **Related PRDs**:
  - `docs/prd/PRD-001-transcript-pipeline.md` (core pipeline)
- **Related RFCs**:
  - `docs/rfc/RFC-025-test-metrics-and-health-tracking.md` (metrics collection - prerequisite)
  - `docs/rfc/RFC-024-test-execution-optimization.md` (test execution optimization)
- **Related Documents**:
  - `docs/TESTING_STRATEGY.md` - Overall testing strategy and test categories
  - `docs/DEVELOPMENT_NOTES.md` - Development workflow and testing requirements
  - `.github/workflows/python-app.yml` - CI test jobs

## Abstract

This RFC defines a strategy for **consuming and visualizing test metrics** to enable quick deviation detection and trend analysis. The strategy focuses on:

1. **Easy access**: Multiple consumption methods (browser, API, PR checks)
2. **Quick detection**: Identify deviations in **< 60 seconds**
3. **Visual dashboards**: Human-readable charts and alerts
4. **Machine-readable API**: JSON endpoints for automation
5. **Zero infrastructure**: Uses GitHub Pages (free, no setup)

**Key Principle:** Metrics are only valuable if they can be consumed quickly. Enable < 60 second deviation detection through multiple access patterns.

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

## Solution: GitHub Pages Metrics Dashboard

**Approach:** Publish metrics to GitHub Pages as both human-readable dashboard and machine-readable JSON.

**Benefits:**

- ✅ **Always accessible**: Public URL (e.g., `https://chipi.github.io/podcast_scraper/metrics/`)
- ✅ **No authentication**: Anyone can view metrics
- ✅ **Auto-updated**: Metrics published after each CI run
- ✅ **Quick consumption**: View dashboard in browser (< 10 seconds)
- ✅ **Machine-readable**: JSON API for automation
- ✅ **Historical trends**: Visual charts showing deviations
- ✅ **Zero infrastructure**: Uses GitHub Pages (free, no setup)

## Implementation Strategy

### 1. Metrics JSON API (Machine-Readable)

**Location:** `https://chipi.github.io/podcast_scraper/metrics/latest.json`

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
```
**Consumption:**

```bash
# Quick check (< 5 seconds)
curl -s https://chipi.github.io/podcast_scraper/metrics/latest.json | jq '.metrics.runtime.total'

# Check for regressions
curl -s https://chipi.github.io/podcast_scraper/metrics/latest.json | jq '.alerts[]'
```
### 2. HTML Dashboard (Human-Readable)

**Location:** `https://chipi.github.io/podcast_scraper/metrics/`

**Features:**

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
```
### 3. Historical Data (JSONL Format)

**Location:** `https://chipi.github.io/podcast_scraper/metrics/history.jsonl`

**Format:** One JSON object per line (JSONL)

```jsonl
{"timestamp":"2024-12-28T19:00:00Z","commit":"abc123","runtime":35.2,"coverage":65.1,"passed":248}
{"timestamp":"2024-12-28T20:00:00Z","commit":"def456","runtime":35.7,"coverage":65.3,"passed":248}
```
**Benefits:**

- Easy to parse line-by-line
- Efficient for streaming
- Can append without rewriting entire file

### 4. GitHub Actions Integration

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

    # Publish to gh-pages branch
    git checkout gh-pages || git checkout --orphan gh-pages
    git add metrics/
    git commit -m "Update metrics: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git push origin gh-pages
```
## Quick Deviation Detection (< 60 seconds)

### Method 1: Browser Dashboard (10 seconds)

1. Open `https://chipi.github.io/podcast_scraper/metrics/`
2. View alerts section (red/yellow highlights)
3. Check trend charts for spikes

### Method 2: JSON API (5 seconds)

```bash
# Check latest metrics
curl -s https://chipi.github.io/podcast_scraper/metrics/latest.json | jq '.alerts'

# Compare with previous run
curl -s https://chipi.github.io/podcast_scraper/metrics/latest.json | jq '.trends'
```
### Method 3: GitHub Actions Job Summary (0 seconds)

- View directly in PR checks
- Shows key metrics and alerts
- No external access needed

### Method 4: Automated Alerts (0 seconds)

- GitHub Actions can comment on PRs with metric changes
- Slack/Discord webhooks for significant deviations
- Email notifications (optional)

## Deviation Detection Logic

### Thresholds

- **Minor deviation**: 5-10% change
- **Significant deviation**: > 10% change
- **Critical deviation**: > 20% change

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

    return alerts
```
## Implementation Plan

### Phase 1: Basic JSON API (1-2 days)

- Generate `metrics/latest.json` from CI
- Publish to gh-pages branch
- Simple deviation detection

**Deliverables:**

- `scripts/generate_metrics.py` - Extract metrics from JUnit/coverage
- GitHub Actions step to publish to gh-pages
- `metrics/latest.json` accessible via GitHub Pages

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

### Phase 4: Automated Alerts (1-2 days)

- PR comments on metric changes
- Webhook notifications (optional)
- Email alerts (optional)

**Deliverables:**

- GitHub Actions step to comment on PRs
- Alert generation logic
- Optional webhook integration

## Access Patterns

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

