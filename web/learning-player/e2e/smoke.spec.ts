import { expect, test } from '@playwright/test'

/**
 * Smoke — REAL API over the COMMITTED validation corpus
 * (tests/fixtures/app-validation-corpus/v3, built by scripts/build_app_validation_corpus.py),
 * NO mocks. Signed-out: Home renders (discover hero + What's-new from the real backend) and
 * sign-in is offered.
 *
 * The "What's new" feature episode is the newest in the corpus by publish date — "Risk Is a
 * Systems Property" (fixture p09_e04, 2026-07-16), the most recent on the #1148 2024→now schedule.
 */
test('Home renders against the real backend (signed out)', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Learning Player')).toBeVisible()
  await expect(page.getByText("Find any moment you've heard.")).toBeVisible() // discover hero
  await expect(page.getByText('Risk Is a Systems Property').first()).toBeVisible() // What's new (newest)
  await expect(page.getByRole('link', { name: 'Sign in' })).toBeVisible()
})
