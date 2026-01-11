# RFC-043: Automated Metrics Alerts

- **Status**: 📝 Draft
- **Related ADRs**:
  - [ADR-042: Proactive Metric Regression Alerting](../adr/ADR-042-proactive-metric-regression-alerting.md)
- **Authors**:
- **Created**: 2026-01-07
- **Related Issues**: #216
- **Related RFCs**:
  - RFC-026: Metrics Consumption and Dashboards (prerequisite - completed)
  - RFC-025: Test Metrics and Health Tracking (prerequisite - completed)
- **Extracted From**: RFC-026 Phase 4 (moved to standalone RFC for independent evolution)

## Abstract

This RFC defines automated alerting for test metrics, extracted from RFC-026 Phase 4 to enable independent evolution. While RFC-026 (Phases 0-3) focused on metrics collection, dashboards, and visualization, this RFC addresses **proactive notification** of metric deviations through PR comments and webhooks.

**Key Principle:** Metrics should alert developers proactively, not require manual checking.

## Background

### Why Extract from RFC-026?

RFC-026 originally included 4 phases:

- ✅ Phase 0: Core Goals (completed)
- ✅ Phase 1: Job Summaries (completed)
- ✅ Phase 2: GitHub Pages JSON API (completed)
- ✅ Phase 3: Visual Dashboards (completed)
- ⏸️ Phase 4: Automated Alerts (NOT started - extracted to this RFC)

**Rationale for extraction:**

1. **Phases 0-3 are complete** and can be marked as done
2. **Phase 4 is substantial** (~1 day effort, non-trivial implementation)
3. **Independent evolution** - Alerts can evolve separately from dashboards
4. **Clear milestone** - RFC-026 can be closed, this RFC tracks remaining work

### Current State (RFC-026 Completion)

Already implemented from RFC-026:

| Component | Status |
| ----------- | -------- |
| Job summaries (PR + nightly) | ✅ Extensive coverage |
| `metrics/latest.json` on GitHub Pages | ✅ Working |
| `metrics/history.jsonl` tracking | ✅ Working |
| Unified dashboard with data source selector | ✅ Working |
| `scripts/dashboard/generate_metrics.py` | ✅ Created |
| `scripts/dashboard/generate_dashboard.py` | ✅ Created |
| Metric alerts in nightly summary | ✅ Working |

**What's missing (this RFC):**

- ❌ PR comments on metric changes
- ❌ Webhook notifications

## Problem Statement

### Current Workflow (Manual Checking)

Developers must **manually** check for metric regressions:

1. Open PR checks
2. Click "View job summary"
3. Scroll to metrics section
4. Compare metrics mentally vs baseline
5. Decide if degradation is acceptable

**Pain Points:**

- Requires extra steps (not in natural PR review flow)
- Easy to miss/ignore
- No comparison to baseline without mental math
- Silent failures (no notification if metrics degrade)

### Desired Workflow (Proactive Alerts)

Metrics should **alert developers automatically**:

1. PR opened
2. Tests run, metrics collected
3. **Comment posted automatically** showing comparison
4. Developer sees alert in PR thread (no extra clicks)
5. For main branch regressions, **webhook alert** (optional)

**Benefits:**

- Zero extra steps
- Impossible to miss
- Clear comparison table
- Proactive notification

## Goals

### Primary Goal

**Proactive metric alerting** with:

- PR comments showing metric comparison (PR vs main baseline)
- Webhook notifications for main branch regressions (optional)
- Clear, actionable alerts with recommendations

### Non-Goals

- Email alerts (out of scope - webhooks are sufficient)
- Blocking CI on metric regressions (alerts are informational only)
- Real-time monitoring (GitHub Actions trigger is sufficient)

## Proposed Solution

### 1. PR Comments (High Priority)

**Goal:** Post comparison comment on every PR showing how metrics changed vs baseline.

#### PR Comment Script Implementation

**Script:** `scripts/generate_pr_comment.py`

```python
#!/usr/bin/env python3
"""Generate PR comment comparing PR metrics vs baseline."""

def generate_pr_comment(pr_metrics: dict, baseline_metrics: dict) -> str:
    """Generate markdown comment comparing PR vs baseline.

    Args:
        pr_metrics: Metrics from this PR
        baseline_metrics: Metrics from main branch

    Returns:
        Markdown formatted comment
    """
    # Calculate deltas
    runtime_delta = pr_metrics["runtime"] - baseline_metrics["runtime"]
    coverage_delta = pr_metrics["coverage"] - baseline_metrics["coverage"]
    tests_delta = pr_metrics["test_count"] - baseline_metrics["test_count"]

    # Generate alerts
    alerts = []
    if abs(runtime_delta / baseline_metrics["runtime"]) > 0.10:

```text

        alerts.append(f"⚠️ Runtime changed by {runtime_delta:+.1f}s ({runtime_delta/baseline_metrics['runtime']*100:+.1f}%)")
    if coverage_delta < -1.0:
        alerts.append(f"⚠️ Coverage dropped by {abs(coverage_delta):.1f}%")

```yaml

### Changes in this PR:

| Metric | Main | This PR | Change |
| -------- | ------ | --------- | -------- |
| Runtime | {baseline_metrics['runtime']:.1f}s | {pr_metrics['runtime']:.1f}s | {runtime_delta:+.1f}s |
| Coverage | {baseline_metrics['coverage']:.1f}% | {pr_metrics['coverage']:.1f}% | {coverage_delta:+.1f}% |
| Tests | {baseline_metrics['test_count']} | {pr_metrics['test_count']} | {tests_delta:+d} |
"""

```text

    if alerts:
        comment += "\n### Alerts:\n" + "\n".join(alerts)

```

#### PR Comment Workflow Integration

Add to `.github/workflows/python-app.yml` (PR jobs):

```yaml

- name: Fetch baseline metrics from main
  if: github.event_name == 'pull_request' && always()

  run: |
    git fetch origin main:main
    BASELINE=$(git show main:metrics/latest.json 2>/dev/null || echo "{}")
    echo "$BASELINE" > metrics/baseline.json

- name: Generate PR comment
  if: github.event_name == 'pull_request' && always()

  run: |
    python scripts/generate_pr_comment.py \
      --pr-metrics metrics/latest.json \
      --baseline metrics/baseline.json \
      --output pr-comment.md

- name: Post PR comment
  if: github.event_name == 'pull_request' && always()

  uses: actions/github-script@v7
  with:
    script: |
      const fs = require('fs');
      if (fs.existsSync('pr-comment.md')) {
        const comment = fs.readFileSync('pr-comment.md', 'utf8');

        // Find existing comment
        const { data: comments } = await github.rest.issues.listComments({
          issue_number: context.issue.number,
          owner: context.repo.owner,
          repo: context.repo.repo,
        });

        const botComment = comments.find(c =>
          c.user.login === 'github-actions[bot]' &&
          c.body.includes('📊 Test Metrics Comparison')
        );

        // Update or create
        if (botComment) {
          await github.rest.issues.updateComment({
            comment_id: botComment.id,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
        } else {
          await github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
        }
      }

```

#### Example Comment Format

```markdown

## 📊 Test Metrics Comparison

### Changes in this PR:

| Metric | Main | This PR | Change |
| -------- | ------ | --------- | -------- |
| Runtime | 35.2s | 37.5s | +2.3s |
| Coverage | 65.3% | 64.8% | -0.5% |
| Tests | 248 | 251 | +3 |
| Flaky Tests | 0 | 2 | +2 |

### Alerts:

⚠️ Runtime increased by 6.5% compared to main branch
⚠️ 2 flaky tests detected (tests that passed on retry)

### Recommendations:
- Review slowest tests if runtime increased significantly
- Investigate flaky tests to improve test stability
- Check if new tests are properly optimized

---
<details>
<summary>📈 View full metrics dashboard</summary>

https://chipi.github.io/podcast_scraper/metrics/
</details>

```

### 2. Webhook Alerts (Medium Priority)

**Goal:** Send Slack/Discord notifications for critical regressions on main branch.

#### Webhook Script Implementation

**Script:** `scripts/send_webhook_alert.py`

```python

#!/usr/bin/env python3
"""Send webhook alerts for metric regressions."""

import requests
import json
from typing import Optional

def send_slack_alert(webhook_url: str, alerts: list, metrics: dict) -> None:
    """Send formatted alert to Slack webhook."""
    payload = {
        "text": "🚨 Test Metrics Alert",
        "attachments": [
            {
                "color": "danger" if alert["severity"] == "error" else "warning",
                "title": alert["metric"],
                "text": alert["message"],
                "fields": [
                    {"title": "Workflow", "value": metrics.get("workflow_name", "Unknown"), "short": True},
                    {"title": "Branch", "value": metrics.get("branch", "main"), "short": True},
                ],
            }
            for alert in alerts
        ],
    }
    requests.post(webhook_url, json=payload)

def send_discord_alert(webhook_url: str, alerts: list, metrics: dict) -> None:

```text

    """Send formatted alert to Discord webhook."""
    embed_color = 0xFF0000 if any(a["severity"] == "error" for a in alerts) else 0xFFA500

```json

    payload = {
        "content": "🚨 Test Metrics Alert",
        "embeds": [
            {
                "title": alert["metric"],
                "description": alert["message"],
                "color": embed_color,
                "fields": [
                    {"name": "Workflow", "value": metrics.get("workflow_name", "Unknown"), "inline": True},
                    {"name": "Branch", "value": metrics.get("branch", "main"), "inline": True},
                ],
            }
            for alert in alerts
        ],
    }
    requests.post(webhook_url, json=payload)

```

#### Webhook Workflow Integration

Add to main branch workflows (`.github/workflows/python-app.yml`, `nightly.yml`):

```yaml

- name: Send webhook alert
  if: always() && github.ref == 'refs/heads/main'

  env:
    WEBHOOK_URL: ${{ secrets.METRICS_WEBHOOK_URL }}
    WEBHOOK_TYPE: ${{ secrets.METRICS_WEBHOOK_TYPE }}  # 'slack' or 'discord'
  run: |
    if [ -n "$WEBHOOK_URL" ] && [ -f metrics/latest.json ]; then
      python scripts/send_webhook_alert.py \
        --webhook-url "$WEBHOOK_URL" \
        --webhook-type "${WEBHOOK_TYPE:-slack}" \
        --metrics metrics/latest.json \
        --threshold error  # Only critical alerts
    fi

```yaml

#### Configuration

Repository secrets to add:
- `METRICS_WEBHOOK_URL` - Slack/Discord webhook URL (optional)
- `METRICS_WEBHOOK_TYPE` - `slack` or `discord` (defaults to `slack`)

## Alert Thresholds

| Metric | Threshold | Severity | Action |
| -------- | ----------- | ---------- | -------- |
| Runtime increase | > 10% | Warning | PR comment |
| Runtime increase | > 20% | Error | PR comment + webhook (main only) |
| Coverage drop | > 1% | Error | PR comment + webhook (main only) |
| Test failures | > 0 | Error | Already in job summary |
| Flaky tests | > 0 | Warning | PR comment |
| Flaky tests | > 5 | Error | PR comment + webhook (main only) |

**Rationale:**
- **Runtime**: 10% warning gives early signal, 20% error for serious regressions
- **Coverage**: 1% drop is significant (can indicate untested code paths)
- **Flaky tests**: Any flakiness is concerning, >5 indicates systemic issue

## Implementation Plan

### Phase 1: PR Comments (Priority 1)

**Deliverables:**
1. `scripts/generate_pr_comment.py` - Generate comparison markdown
2. Update `.github/workflows/python-app.yml` - Add PR comment steps
3. Test with sample PR

**Effort:** 2-3 hours

### Phase 2: Webhook Alerts (Priority 2)

**Deliverables:**
1. `scripts/send_webhook_alert.py` - Send alerts to webhooks
2. Update workflows (python-app.yml, nightly.yml) - Add webhook steps
3. Documentation for secret configuration

**Effort:** 1-2 hours

### Phase 3: Testing & Refinement

**Deliverables:**
1. Test PR comments don't duplicate
2. Test webhook alerts (if configured)
3. Refine alert thresholds based on feedback
4. Update documentation

**Effort:** 1-2 hours

**Total Effort:** ~1 day

## Acceptance Criteria

- [ ] PR comments posted when metrics deviate significantly
- [ ] Comments show comparison table (main vs PR)
- [ ] Comments are updated (not duplicated) when PR is updated
- [ ] Webhook alerts sent for critical regressions (if secret configured)
- [ ] Alerts are informational only (don't block CI)
- [ ] Documentation updated with webhook configuration

## Documentation

### Files to Update

1. `docs/ci/METRICS.md` - Add PR comment and webhook sections
2. `docs/ci/WORKFLOWS.md` - Document alert workflow steps
3. `docs/rfc/RFC-026-metrics-consumption-and-dashboards.md` - Update to reference RFC-040 for Phase 4
4. `README.md` - Optional: Add note about PR metric comments

### Example Documentation

**docs/ci/METRICS.md:**

```markdown

## Automated Alerts

### PR Comments

Every PR automatically receives a comment comparing metrics vs the main branch:
- Runtime changes
- Coverage changes
- Test count changes
- Flaky test detection

The comment is updated (not duplicated) when the PR is updated.

### Webhook Alerts (Optional)

For main branch regressions, webhook alerts can be sent to Slack/Discord.

**Configuration:**
1. Create a webhook URL in Slack/Discord
2. Add to repository secrets:
   - `METRICS_WEBHOOK_URL` - Your webhook URL
   - `METRICS_WEBHOOK_TYPE` - `slack` or `discord`
3. Alerts will be sent for critical regressions on main branch

```

## Future Enhancements

Potential future work (out of scope for this RFC):

- Email alerts
- Customizable thresholds per repository
- Alert muting/snoozing
- Integration with other services (PagerDuty, etc.)
- Historical alert tracking

## References

- RFC-026: Metrics Consumption and Dashboards
- RFC-025: Test Metrics and Health Tracking
- Issue #216: Implement automated alerts
- `docs/ci/METRICS.md` - Current metrics documentation
