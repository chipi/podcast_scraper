import { expect, test } from '@playwright/test'

/**
 * Smoke — REAL API, REAL fixture corpus, NO mocks. Signed-out: the shell renders, the
 * catalog lists the fixture episode (served by the actual backend), and sign-in is offered.
 */
test('app shell + catalog render against the real backend (signed out)', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Learning Player')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Your Library' })).toBeVisible()
  // Episode comes from the real API serving e2e/fixtures/corpus.
  await expect(page.getByText('How Sleep Consolidates Memory')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Sign in' })).toBeVisible()
})
