# RFC-026 Phase 4: Automated Alerts - Implementation Plan

## Overview

Phase 4 adds automated alerting for metric regressions and changes. The goal is to proactively notify developers about test health issues without requiring manual checks.

## Design Principles

1. **Informational Only** - Alerts don't block merges or fail CI
2. **Actionable** - Alerts include context and suggestions
3. **Configurable** - Can be enabled/disabled per team
4. **Low Noise** - Only alert on significant changes (> 10% for most metrics)
5. **Multiple Channels** - Support PR comments, webhooks, and optional email

---

## Implementation Approach

### Option A: PR-Based Alerts (Recommended First Step)

**When:** On PR creation/updates (when PR has test results)

**What:**

- Compare PR test metrics with main branch baseline
- Comment on PR if significant deviations detected
- Show trend comparison (PR vs main)

**Pros:**

- Immediate feedback for PR authors
- Contextual (shows what changed in the PR)
- No external dependencies

**Cons:**

- Requires PR to have test results (only on PRs that run tests)
- May not catch regressions that only appear on main

**Implementation:**

1. Add GitHub Actions step to PR workflow
2. Fetch baseline metrics from `metrics/latest.json` (from main branch)
3. Compare PR test results with baseline
4. Generate comment with alerts
5. Post comment to PR using GitHub API

---

### Option B: Main Branch Alerts (Recommended Second Step)

**When:** On push to main (after slow tests complete)

**What:**

- Compare current main metrics with historical baseline
- Post comment to commit or create issue if critical regression
- Send webhook notification for significant changes

**Pros:**

- Catches regressions that only appear on main
- Can trigger on slow tests that don't run in PRs
- Historical comparison (vs last 5-10 runs)

**Cons:**

- Less immediate (runs after merge)
- May create noise if metrics fluctuate

**Implementation:**

1. Add step to main branch workflow (after slow tests)
2. Load history and compare with current run
3. Generate alerts using existing `detect_deviations()` logic
4. Post to commit comments or create GitHub issue
5. Send webhook if configured

---

### Option C: Nightly Summary (Recommended Third Step)

**When:** After nightly workflow completes

**What:**

- Weekly summary of metrics trends
- Highlight significant changes over the week
- Optional email digest

**Pros:**

- Low noise (weekly summary)
- Good for maintainers to track long-term trends
- Can include recommendations

**Cons:**

- Less immediate
- May miss urgent issues

**Implementation:**

1. Add step to nightly workflow
2. Analyze week's metrics (last 7 runs)
3. Generate summary report
4. Post to GitHub issue or send email

---

## Recommended Implementation Order

### Step 1: PR Comments (1 day)

**Goal:** Alert PR authors about metric changes in their PR

**Implementation:**

1. **Create `scripts/generate_pr_comment.py`:**

   ```python
   def generate_pr_comment(pr_metrics, baseline_metrics):
       """Generate markdown comment comparing PR vs baseline."""
       # Compare metrics
       # Generate alerts
       # Format as markdown
       return comment_markdown
   ```

2. **Add to PR workflow** (`.github/workflows/python-app.yml`):

   ```yaml
   - name: Generate PR metrics comment
     if: github.event_name == 'pull_request' && always()
     run: |
       # Fetch baseline from main branch
       git fetch origin main:main
       BASELINE=$(git show main:metrics/latest.json 2>/dev/null || echo "{}")

       # Generate current PR metrics
       python scripts/generate_metrics.py \
         --reports-dir reports \
         --output metrics/pr-metrics.json

       # Generate comment
       python scripts/generate_pr_comment.py \
         --pr-metrics metrics/pr-metrics.json \
         --baseline "$BASELINE" \
         --output pr-comment.md

       # Post comment using GitHub CLI or API
   ```

3. **Post comment using GitHub CLI:**

   ```yaml
   - name: Post PR comment
     uses: actions/github-script@v7
     with:
       script: |
         const comment = fs.readFileSync('pr-comment.md', 'utf8');
         github.rest.issues.createComment({
           issue_number: context.issue.number,
           owner: context.repo.owner,
           repo: context.repo.repo,
           body: comment
         });
   ```

**Comment Format:**

```markdown
## 📊 Test Metrics Comparison

### Changes in this PR:
- **Runtime**: +2.3s (15% increase) ⚠️
- **Coverage**: -0.5% (minor decrease)
- **Test Count**: +3 (new tests added)

### Alerts:
⚠️ **Runtime Regression**: Runtime increased by 15% compared to main branch baseline

### Recommendations:
- Review slowest tests to identify bottlenecks
- Consider optimizing test setup/teardown
```

---

### Step 2: Webhook Notifications (1 day)

**Goal:** Send alerts to Slack/Discord for critical regressions

**Implementation:**

1. **Create `scripts/send_webhook_alert.py`:**

   ```python
   def send_webhook_alert(webhook_url, alerts, metrics):
       """Send formatted alert to webhook."""
       payload = {
           "text": "Test Metrics Alert",
           "attachments": [
               {
                   "color": "danger" if severity == "error" else "warning",
                   "title": alert["metric"],
                   "text": alert["message"],
                   "fields": [
                       {"title": "Runtime", "value": f"{metrics['runtime']:.1f}s"},
                       {"title": "Coverage", "value": f"{metrics['coverage']:.1f}%"},
                   ]
               }
           ]
       }
       requests.post(webhook_url, json=payload)
   ```

2. **Add to workflow:**

   ```yaml
   - name: Send webhook alert
     if: always() && github.ref == 'refs/heads/main'
     env:
       WEBHOOK_URL: ${{ secrets.METRICS_WEBHOOK_URL }}
     run: |
       if [ -n "$WEBHOOK_URL" ] && [ -f metrics/latest.json ]; then
         python scripts/send_webhook_alert.py \
           --webhook-url "$WEBHOOK_URL" \
           --metrics metrics/latest.json \
           --threshold error  # Only send critical alerts
       fi
   ```

3. **Configuration:**
   - Add `METRICS_WEBHOOK_URL` to repository secrets
   - Support Slack, Discord, or generic webhooks
   - Only send for severity "error" (not warnings)

**Webhook Format (Slack):**

```json
{
  "text": "🚨 Test Metrics Alert",
  "attachments": [{
    "color": "danger",
    "title": "Runtime Regression",
    "text": "Runtime increased by 20% compared to last 5 runs",
    "fields": [
      {"title": "Current", "value": "45.2s"},
      {"title": "Baseline", "value": "37.6s"},
      {"title": "Commit", "value": "abc123"},
      {"title": "Workflow", "value": "https://github.com/..."}
    ]
  }]
}
```

---

### Step 3: Email Alerts (Optional, Future)

**Goal:** Weekly digest email for maintainers

**Implementation:**

- Use GitHub Actions email action or SendGrid
- Generate weekly summary from history
- Send only if significant changes detected
- Include dashboard link

**Priority:** Low - Webhooks are usually sufficient

---

## Technical Details

### Alert Thresholds

**When to Alert:**

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Runtime increase | > 10% | Warning |
| Runtime increase | > 20% | Error |
| Coverage drop | > 1% | Error |
| Test count change | > 5% | Info |
| Flaky test increase | > 50% | Warning |
| Test failures | > 0 | Error |

### Comparison Methods

**PR Comments:**

- Compare PR metrics vs main branch baseline
- Show absolute and percentage changes
- Highlight regressions

**Main Branch Alerts:**

- Compare current vs last 5 runs (median for runtime)
- Compare current vs last 10 runs (average for coverage)
- Use existing `detect_deviations()` logic

### Comment Management

**Avoid Duplicate Comments:**

- Check if comment already exists (search for bot username)
- Update existing comment instead of creating new
- Use GitHub API to find/update comments

**Comment Formatting:**

- Use markdown for readability
- Include links to workflow runs
- Show visual indicators (✅ ⚠️ 🔴)
- Collapsible sections for details

---

## Files to Create

1. **`scripts/generate_pr_comment.py`**
   - Compare PR metrics with baseline
   - Generate markdown comment
   - Format alerts and recommendations

2. **`scripts/send_webhook_alert.py`**
   - Format alerts for webhooks
   - Support Slack/Discord formats
   - Handle errors gracefully

3. **Update `.github/workflows/python-app.yml`**
   - Add PR comment step
   - Add webhook step (optional, behind secret)

---

## Configuration

### Repository Secrets

- `METRICS_WEBHOOK_URL` - Webhook URL for Slack/Discord (optional)
- `METRICS_EMAIL_API_KEY` - For email alerts (optional, future)

### Workflow Configuration

```yaml
# Enable/disable features
env:
  ENABLE_PR_COMMENTS: true
  ENABLE_WEBHOOK_ALERTS: false  # Requires secret
  ALERT_THRESHOLD: error  # error, warning, or info
```

---

## Success Criteria

- [ ] PR comments posted when metrics deviate significantly
- [ ] Comments are informative and actionable
- [ ] No duplicate comments (update existing)
- [ ] Webhook alerts sent for critical regressions (if configured)
- [ ] Alerts don't block CI or merges
- [ ] Low noise (only significant changes)

---

## Example Workflow

### PR Comment Flow

1. PR created/updated
2. Tests run (fast tests)
3. Generate PR metrics
4. Fetch baseline from main
5. Compare and detect deviations
6. Generate comment
7. Post to PR (or update existing)

### Main Branch Alert Flow

1. PR merged to main
2. Slow tests run
3. Generate metrics
4. Compare with history
5. Detect deviations
6. If critical (error severity):
   - Post commit comment
   - Send webhook (if configured)
7. If warning:
   - Include in job summary only

---

## Future Enhancements

1. **Issue Creation** - Create GitHub issue for critical regressions
2. **Auto-Revert** - Option to auto-revert commits with severe regressions
3. **Team Notifications** - Route alerts to specific teams
4. **Custom Thresholds** - Per-metric threshold configuration
5. **Alert Suppression** - Allow suppressing alerts for known issues

---

## Estimated Time

- **Step 1 (PR Comments)**: 1 day
- **Step 2 (Webhooks)**: 1 day
- **Step 3 (Email)**: 1 day (optional)

**Total**: 2-3 days (depending on optional features)

---

## Recommendation

**Start with Step 1 (PR Comments)** - This provides immediate value with minimal complexity. Add webhooks later if needed.

**Skip email alerts** - Webhooks are usually sufficient, and email adds complexity.
