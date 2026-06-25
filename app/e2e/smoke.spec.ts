import { expect, test } from '@playwright/test'

/**
 * Smoke — REAL API, REAL fixture corpus, NO mocks. Signed-out: Home renders (discover hero +
 * What's new from the real backend) and sign-in is offered.
 */
test('Home renders against the real backend (signed out)', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Learning Player')).toBeVisible()
  await expect(page.getByText("Find any moment you've heard.")).toBeVisible() // discover hero
  await expect(page.getByText('Attention and Deep Work').first()).toBeVisible() // What's new (newest)
  await expect(page.getByRole('link', { name: 'Sign in' })).toBeVisible()
})
