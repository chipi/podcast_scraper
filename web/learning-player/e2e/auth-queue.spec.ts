import { expect, test } from '@playwright/test'

/**
 * Real auth + queue — REAL API over the COMMITTED validation corpus, NO mocks. Drives the actual
 * mock-OAuth sign-in flow (login → callback → session, same-origin via the preview proxy), then
 * exercises the auth-gated queue end-to-end (add from a card → it appears in the queue view,
 * persisted through the real API). Per-user state is written to the gitignored APP_DATA_DIR, so
 * the committed corpus tree is never mutated.
 *
 * The queued episode is a real corpus one: "Index Investing Without the Myths"
 * ("Long Horizon Notes" / fixture p05).
 */
test('sign in (mock OAuth), add to queue, see it in the queue view', async ({ page }, testInfo) => {
  await page.goto('/')

  // Real sign-in: Sign in link → login page → the dev picker (mock provider) → sign in as a
  // custom identity, which drives the actual OAuth flow (login → callback → session). The id
  // is UNIQUE PER PROJECT (mobile-chrome / desktop-chrome) so the two projects, which run in
  // parallel, get SEPARATE queues — sharing one user meant both mutated one queue file
  // concurrently (a read-modify-write race that intermittently dropped the write → "queue
  // empty"). The other auth specs already isolate per test via signInIsolated; this matches.
  await page.getByRole('link', { name: 'Sign in' }).click()
  await page.getByTestId('dev-custom-input').fill(`queue-user-${testInfo.project.name}`)
  await page.getByTestId('dev-custom-submit').click()

  // Back signed-in: the header now offers Sign out (auth-gated nav rehydrated).
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()

  // Add a SPECIFIC episode to the queue from its catalog card (auth-gated control). Idempotent
  // (only ever ADD, never toggle off): click "Add to queue" only if it isn't already queued,
  // then confirm the queued state — robust to a retry re-running against an already-queued item.
  await page.goto('/podcast/p05') // #1148: show page lists all its episodes
  const card = page.locator('article').filter({ hasText: 'Index Investing Without the Myths' })
  // Wait for the auth-gated queue control to render (the session rehydrates after the full
  // page reload). Match either label so we can branch idempotently below.
  const queueBtn = card.getByRole('button', { name: /queue/i })
  await expect(queueBtn).toBeVisible()
  if ((await queueBtn.getAttribute('aria-label')) === 'Add to queue') {
    // The button flips optimistically (store state), so it does NOT prove the write landed.
    // Wait for the PUT /api/app/queue to actually persist (2xx, atomic write) before reading
    // the queue view — otherwise, under load, the GET can race ahead of the write ("empty").
    const putPersisted = page.waitForResponse(
      (r) => r.url().endsWith('/api/app/queue') && r.request().method() === 'PUT' && r.ok(),
    )
    await queueBtn.click()
    await putPersisted
  }
  await expect(card.getByRole('button', { name: 'Remove from queue' })).toBeVisible()

  // The queue view (auth-gated route) lists the queued episode, served by the real API.
  await page.goto('/queue')
  await expect(page).toHaveURL(/\/queue/)
  await expect(page.getByText('Index Investing Without the Myths').first()).toBeVisible()
})
