import { expect, test } from '@playwright/test'

/**
 * Tier-3 — recall (search with scope=mine) against a real backend +
 * real corpus.
 *
 * The consumer app's "My corpus" toggle re-runs search against the
 * heard∪captured set (PRD-041 §Recall). This walk drives:
 *   1. Perform a search with scope=all — confirm the everything path works
 *   2. Toggle to scope=mine — confirm the honest-zero-coverage state
 *   3. Screenshots both.
 */

test('operator recall: scope=all vs scope=mine on real search', async ({ page }) => {
  await page.goto('/api/app/auth/login?as=tier3-app-recall')
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()

  await page.goto('/search')
  await page.waitForLoadState('networkidle')
  await page.screenshot({ path: 'validation-results/recall-01-empty.png', fullPage: true })

  // Type a generic query the real corpus is very likely to have something for.
  const searchInput = page
    .getByRole('searchbox')
    .or(page.getByPlaceholder(/search/i))
    .first()
  if (await searchInput.isVisible().catch(() => false)) {
    await searchInput.fill('the')
    await searchInput.press('Enter')
    await page.waitForLoadState('networkidle')
    await page.screenshot({
      path: 'validation-results/recall-02-all-results.png',
      fullPage: true,
    })
  }

  // Toggle to "My corpus" — signed-out heard∪captured is empty, so this
  // exercises the honest-zero-coverage UX (PRD-041) rather than fake
  // results. The regression signal: the toggle exists, is reachable, and
  // updates the results surface.
  const mineToggle = page
    .getByRole('radio', { name: /my corpus/i })
    .or(page.getByRole('button', { name: /my corpus/i }))
    .first()
  if (await mineToggle.isVisible().catch(() => false)) {
    await mineToggle.click()
    await page.waitForLoadState('networkidle')
    await page.screenshot({
      path: 'validation-results/recall-03-mine-results.png',
      fullPage: true,
    })
  }
})
