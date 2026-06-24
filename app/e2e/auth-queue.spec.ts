import { expect, test } from '@playwright/test'

/**
 * Real auth + queue — NO mocks. Drives the actual mock-OAuth sign-in flow (login → callback →
 * session, same-origin via the preview proxy), then exercises the auth-gated queue end-to-end
 * (add from a card → it appears in the queue view, persisted through the real API).
 */
test('sign in (mock OAuth), add to queue, see it in the queue view', async ({ page }) => {
  await page.goto('/')

  // Real sign-in: Sign in link → login page → the actual OAuth flow (mock provider).
  await page.getByRole('link', { name: 'Sign in' }).click()
  await page.getByRole('button', { name: 'Sign in' }).click()

  // Back signed-in: the header now offers Sign out + the Queue link.
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()
  await expect(page.getByRole('link', { name: /Queue/ })).toBeVisible()

  // Add the episode to the queue from its catalog card (auth-gated control).
  await page.getByRole('button', { name: 'Add to queue' }).first().click()

  // The queue view (auth-gated route) lists the queued episode, served by the real API.
  await page.getByRole('link', { name: /Queue/ }).click()
  await expect(page).toHaveURL(/\/queue/)
  await expect(page.getByText('How Sleep Consolidates Memory')).toBeVisible()
})
