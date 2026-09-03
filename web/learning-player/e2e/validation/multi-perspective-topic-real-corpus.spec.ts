import { expect, test } from '@playwright/test'

/**
 * Tier-3 — multi-perspective topic card (#1146) against a real corpus.
 *
 * Real-corpus test because the perspectives surface depends on having multiple speakers
 * discussing the same topic across episodes — a synthetic corpus with one speaker per show
 * cannot exercise this meaningfully.
 *
 * This spec used to hunt for "any topic chip visible on Home" and, when it found none,
 * `test.skip` itself with the excuse that "Tier-3 walks are inspection artifacts, not
 * gate-tests". That made it unfalsifiable: it reported a pass-shaped skip on every run and
 * would never have caught the renderer breaking. Home does not surface a topic link on the
 * committed corpus, so it ALWAYS skipped.
 *
 * It now navigates DIRECTLY to a topic the corpus is known to carry and asserts. On the
 * committed app-validation corpus `topic:risk-management` returns 10 perspectives from named
 * guests (verified against /api/app/topics/topic:risk-management/perspectives), so there is
 * nothing conditional left.
 *
 * Signals a regression in the topic-card multi-perspective renderer, NOT in the underlying
 * enrichment envelopes (that's ADR-104 territory).
 */

// Override for an operator run against a different corpus (APP_CORPUS_PATH).
const TOPIC = process.env.TIER3_TOPIC_ID || 'topic:risk-management'

test('operator multi-perspective topic card: open + render', async ({ page }) => {
  await page.goto(`/topic/${encodeURIComponent(TOPIC)}`)
  await page.waitForLoadState('networkidle')
  await page.screenshot({
    path: 'validation-results/perspective-01-topic.png',
    fullPage: true,
  })

  // The section renders only when the topic HAS perspectives, so its presence is the
  // assertion — no `if (visible)` guard, or the test proves nothing.
  const section = page.getByTestId('topic-perspectives')
  await expect(section).toBeVisible()

  // "Multi-perspective" means at least two distinct guests with a take. Asserting >= 2
  // rather than the committed corpus's exact 10 keeps an operator run against their own
  // corpus (APP_CORPUS_PATH) meaningful instead of red for being differently shaped.
  const rows = page.getByTestId('topic-perspective')
  expect(await rows.count()).toBeGreaterThanOrEqual(2)

  // Each perspective must name whose take it is — an unnamed row means the renderer lost
  // its person binding, which is exactly the #1146 regression this walk exists to catch.
  const first = rows.first()
  await expect(first).toBeVisible()
  expect((await first.innerText()).trim().length).toBeGreaterThan(0)

  await page.screenshot({
    path: 'validation-results/perspective-02-perspectives.png',
    fullPage: true,
  })
})
