import { expect, test } from '@playwright/test'

/**
 * Real auth + queue — REAL API over the COMMITTED validation corpus, NO mocks. Drives the actual
 * mock-OAuth sign-in flow (login → callback → session, same-origin via the preview proxy), then
 * exercises the auth-gated queue end-to-end (add from a card → it appears in the queue view,
 * persisted through the real API). Per-user state is written to the gitignored APP_DATA_DIR, so
 * the committed corpus tree is never mutated.
 *
 * The queued episode is a real corpus one: "Real Estate: Numbers Before Narratives"
 * ("Long Horizon Notes" / fixture p05).
 */
test('sign in (mock OAuth), add to queue, see it in the queue view', async ({ page }) => {
  await page.goto('/')

  // Real sign-in: Sign in link → login page → the dev picker (mock provider) → sign in as a
  // custom identity, which drives the actual OAuth flow (login → callback → session).
  await page.getByRole('link', { name: 'Sign in' }).click()
  await page.getByTestId('dev-custom-input').fill('queue-user')
  await page.getByTestId('dev-custom-submit').click()

  // Back signed-in: the header now offers Sign out (auth-gated nav rehydrated).
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()

  // Add a SPECIFIC episode to the queue from its catalog card (auth-gated control).
  // Genuinely idempotent so the two parallel projects (which share one mock-user queue) can't
  // race: we only ever ADD (never toggle off) — click "Add to queue" only if it isn't already
  // queued, then confirm the queued state. The shared queue monotonically keeps the episode.
  await page.goto('/catalog')
  const card = page.locator('article').filter({ hasText: 'Real Estate: Numbers Before Narratives' })
  // Wait for the auth-gated queue control to render (the session rehydrates after the full
  // page reload). Match either label so we can branch idempotently below.
  const queueBtn = card.getByRole('button', { name: /queue/i })
  await expect(queueBtn).toBeVisible()
  if ((await queueBtn.getAttribute('aria-label')) === 'Add to queue') await queueBtn.click()
  await expect(card.getByRole('button', { name: 'Remove from queue' })).toBeVisible()

  // The queue view (auth-gated route) lists the queued episode, served by the real API.
  await page.goto('/queue')
  await expect(page).toHaveURL(/\/queue/)
  await expect(page.getByText('Real Estate: Numbers Before Narratives').first()).toBeVisible()
})
