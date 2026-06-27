import { expect, test } from '@playwright/test'

/**
 * Smoke — REAL API over the COMMITTED validation corpus
 * (tests/fixtures/app-validation-corpus/v2, built by scripts/build_app_validation_corpus.py),
 * NO mocks. Signed-out: Home renders (discover hero + What's-new from the real backend) and
 * sign-in is offered.
 *
 * The "What's new" feature episode is the newest in the corpus — "Index Investing Without the
 * Myths" from the "Long Horizon Notes" show (fixture p05).
 */
test('Home renders against the real backend (signed out)', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Learning Player')).toBeVisible()
  await expect(page.getByText("Find any moment you've heard.")).toBeVisible() // discover hero
  await expect(page.getByText('Index Investing Without the Myths').first()).toBeVisible() // What's new (newest)
  await expect(page.getByRole('link', { name: 'Sign in' })).toBeVisible()
})
