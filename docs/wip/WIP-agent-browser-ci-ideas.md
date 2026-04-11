# WIP: Agent-Browser Loop — CI Enhancement Ideas

**Status:** Ideas for later — pick up when useful
**Parent guide:** [Agent-Browser Closed Loop Guide](../guides/AGENT_BROWSER_LOOP_GUIDE.md)
**Created:** April 2026

---

## Context

The agent-browser closed loop works locally in two modes: automated (`make test-ui-e2e`)
and live co-development (shared Chrome session). The automated mode already runs in CI
as the `viewer-e2e` job. This file collects ideas for extending CI to give the agent
(and humans) richer feedback from CI failures.

---

## Idea 1: Upload Playwright traces and HTML report as CI artifacts

**Effort:** Low (workflow YAML change)
**Value:** High — fastest path to debugging CI-only failures

### What

When the `viewer-e2e` job fails, upload the Playwright trace files and HTML report as
GitHub Actions artifacts. The trace contains DOM snapshots, network requests, console
logs, and action screenshots for every step of the failing test.

### Why

CI failures that don't reproduce locally are the hardest to debug. Today you get a
text log. With trace artifacts, the agent can download the trace and inspect the exact
DOM state, network response, and console output at the point of failure — the same
information it would get from a live DevTools MCP session, but post-mortem.

### Implementation sketch

```yaml
# In .github/workflows/viewer-e2e.yml (or wherever the job lives)
- name: Upload Playwright report on failure
  if: failure()
  uses: actions/upload-artifact@v4
  with:
    name: playwright-report
    path: web/gi-kg-viewer/playwright-report/
    retention-days: 14

- name: Upload Playwright traces on failure
  if: failure()
  uses: actions/upload-artifact@v4
  with:
    name: playwright-traces
    path: web/gi-kg-viewer/test-results/
    retention-days: 14
```

### Agent workflow after CI failure

1. Agent downloads the trace artifact from the failed CI run
2. Opens it locally: `npx playwright show-trace test-results/.../trace.zip`
3. Inspects DOM snapshot, network calls, console at the failure point
4. Diagnoses and fixes — same loop as local automated mode

---

## Idea 2: Console error gate

**Effort:** Low (test fixture change)
**Value:** Medium — catches silent JS errors that tests don't assert on

### What

Add a global `page.on('console')` listener in the Playwright test fixtures that
collects all `console.error` messages during each test. Fail the test if any
unexpected errors were logged, even if all assertions passed.

### Why

A test can pass (the button rendered, the text matched) while a JS error fires in
the background — a failed fetch retry, a Vue reactivity warning, a third-party
library error. These are invisible in the test result but visible in production.

### Implementation sketch

```typescript
// In e2e/fixtures.ts or a shared beforeEach
const consoleErrors: string[] = []
page.on('console', msg => {
  if (msg.type() === 'error') consoleErrors.push(msg.text())
})

// In afterEach
test.afterEach(async ({}, testInfo) => {
  const unexpected = consoleErrors.filter(
    e => !KNOWN_BENIGN_ERRORS.some(pattern => e.includes(pattern))
  )
  if (unexpected.length > 0) {
    testInfo.annotations.push({
      type: 'console-errors',
      description: unexpected.join('\n'),
    })
    // Optionally: expect(unexpected).toHaveLength(0)
  }
  consoleErrors.length = 0
})
```

A `KNOWN_BENIGN_ERRORS` allowlist handles expected noise (e.g., third-party library
warnings that are not actionable).

### Considerations

- Start with annotation-only (warning) before making it a hard fail, to calibrate
  the noise level
- Vue dev-mode warnings are verbose — filter or categorize them separately

---

## Idea 3: Screenshot diff / visual regression

**Effort:** Medium (needs baseline management, tolerance tuning)
**Value:** Medium-high — catches visual regressions that DOM assertions miss

### What

Capture screenshots of key surfaces (Dashboard, Library list, Graph with nodes,
Digest topic bands) and compare them against committed baselines. Fail if pixel
difference exceeds a threshold.

### Why

DOM structure can be correct while CSS breaks the layout — a flexbox overflow, a
z-index collision, a missing theme token. Screenshot diffs catch what accessibility
snapshots and DOM assertions cannot.

### Implementation sketch

Playwright has built-in visual comparison:

```typescript
await expect(page).toHaveScreenshot('dashboard-overview.png', {
  maxDiffPixelRatio: 0.01,
  animations: 'disabled',
})
```

Baselines live in `e2e/*.spec.ts-snapshots/` and are committed to the repo.

### Considerations

- **Font rendering differs across OS** — CI (Linux) vs local (macOS). Use
  `--update-snapshots` on CI to generate Linux baselines; compare only on CI.
  Local runs skip visual comparison or use a Docker container for consistency.
- **Dynamic data** — mock API responses in visual tests to ensure deterministic
  content. The existing mock patterns in `search-to-graph-mocks.spec.ts` are a
  good template.
- **Maintenance cost** — every intentional UI change requires updating baselines.
  Keep the set small: 4-6 key surfaces, not every possible state.
- **Threshold tuning** — anti-aliasing and subpixel rendering cause false positives.
  Start with `maxDiffPixelRatio: 0.02` and tighten over time.

---

## Idea 4: Accessibility audit in E2E

**Effort:** Medium (add axe-core, triage initial violations)
**Value:** Medium — automated a11y regression detection

### What

Run axe-core accessibility checks on key pages as part of the E2E suite. Fail on
new violations above a severity threshold.

### Why

The viewer already uses semantic HTML and ARIA attributes (the `E2E_SURFACE_MAP.md`
documents `role`, `aria-label`, etc.). An automated audit ensures these don't
regress and catches issues that manual testing misses (color contrast, missing
labels on new elements, focus order).

### Implementation sketch

```typescript
import AxeBuilder from '@axe-core/playwright'

test('dashboard has no critical a11y violations', async ({ page }) => {
  await page.goto('/')
  // ... navigate to dashboard state ...

  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa'])
    .analyze()

  expect(results.violations.filter(v => v.impact === 'critical')).toHaveLength(0)
})
```

### Considerations

- **Initial triage** — first run will likely surface existing violations. Fix
  critical/serious ones, add known minor issues to an allowlist, then enforce
  zero-regression going forward.
- **Scope** — run on 3-4 key states (empty shell, loaded graph, Library with
  episodes, Dashboard). Not every possible state.
- **CI time** — axe-core adds ~1-2s per page scan. Negligible in the context of
  E2E test runtime.

---

## Priority suggestion

| Idea | Effort | Impact | Suggested order |
| ---- | ------ | ------ | --------------- |
| 1. Trace/report upload | Low | High | First — immediate debugging value |
| 2. Console error gate | Low | Medium | Second — low effort, catches real bugs |
| 3. Screenshot diffs | Medium | Medium-high | Third — after baseline workflow is clear |
| 4. Accessibility audit | Medium | Medium | Fourth — after initial a11y triage |
