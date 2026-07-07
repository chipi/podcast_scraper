import { expect, test } from '@playwright/test'

/**
 * Tier-3 — multi-perspective topic card (#1146) against a real corpus.
 *
 * Real-corpus test because the perspectives surface depends on having
 * multiple speakers discussing the same topic across episodes — a
 * synthetic corpus with one speaker per show cannot exercise this
 * meaningfully. Operator-driven `APP_CORPUS_PATH` runs light this up.
 *
 * Signals a regression in the topic-card multi-perspective renderer,
 * NOT in the underlying enrichment envelopes (that's ADR-104 territory).
 */

test('operator multi-perspective topic card: open + render', async ({ page }) => {
  // Home renders — pick any topic chip / entity link visible on the surface.
  await page.goto('/')
  await page.waitForLoadState('networkidle')
  await page.screenshot({
    path: 'validation-results/perspective-01-home.png',
    fullPage: true,
  })

  // Look for a topic surface — either the "Trending topics" chip row or an
  // in-catalog topic link. Real corpora surface these via /api/app/corpus/enrichment.
  const topicLink = page
    .locator('[data-testid="topic-card"], [data-testid="topic-chip"], a[href*="/topic/"]')
    .first()
  if (!(await topicLink.isVisible().catch(() => false))) {
    // Skip is legitimate if the corpus doesn't have a topic surface —
    // Tier-3 walks are inspection artifacts, not gate-tests.
    test.skip(true, 'Real corpus has no topic surface visible on Home')
    return
  }

  await topicLink.click()
  await page.waitForLoadState('networkidle')
  await page.screenshot({
    path: 'validation-results/perspective-02-topic-card.png',
    fullPage: true,
  })

  // Multi-perspective section — present when the topic has speakers with
  // divergent takes. Absence here is not a bug for every corpus.
  const perspectiveSection = page
    .getByText(/perspective|takes|speakers|contrasting/i)
    .first()
  if (await perspectiveSection.isVisible().catch(() => false)) {
    await page.screenshot({
      path: 'validation-results/perspective-03-multi-view.png',
      fullPage: true,
    })
  }
})
