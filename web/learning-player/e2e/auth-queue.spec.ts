import { expect, test } from '@playwright/test'
import { expectSignedIn } from './helpers'

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
  // RFC-120: / redirects to /welcome, which carries both a landing-cta-signin link and the masthead
  // Sign in link — use the landing's dedicated testid to avoid the strict-mode 2-element violation.
  await page.getByTestId('landing-cta-signin').click()
  await page.getByTestId('dev-custom-input').fill(`queue-user-${testInfo.project.name}`)
  await page.getByTestId('dev-custom-submit').click()

  // Back signed-in: the header now offers Sign out (auth-gated nav rehydrated).
  await expectSignedIn(page)

  // Add a SPECIFIC episode to the queue from its catalog card (auth-gated control). Idempotent
  // (only ever ADD, never toggle off): click "Add to queue" only if it isn't already queued,
  // then confirm the queued state — robust to a retry re-running against an already-queued item.
  // Registered BEFORE the navigation, because `waitForResponse` only sees responses that arrive
  // after it is set up — and the queue GET fires during App.vue's mount, which is over long before
  // any locator below resolves.
  const queueHydrated = page
    .waitForResponse((r) => /\/api\/app\/queue(\?|$)/.test(r.url()) && r.request().method() === 'GET')
    .catch(() => null)
  await page.goto('/podcast/p05') // #1148: show page lists all its episodes
  await queueHydrated

  const card = page.locator('article').filter({ hasText: 'Index Investing Without the Myths' })
  // Wait for the auth-gated queue control to render (the session rehydrates after the full
  // page reload). Match either label so we can branch idempotently below.
  const queueBtn = card.getByRole('button', { name: /queue/i })
  await expect(queueBtn).toBeVisible()
  // Reading the label is only meaningful once the store behind it has hydrated — hence the wait
  // above. The button paints from an empty queue first, so a read that beat the GET always saw
  // "Add to queue". When the item was in fact ALREADY queued (any second run against the same api
  // container), the label then flipped to "Remove from queue" under the branch: the click sent a
  // DELETE while this spec sat waiting for a POST that would never come, and burned the full
  // 60-second timeout. It looked like the api was too slow to answer. Nothing was slow — the click
  // did the opposite of what the branch had decided.
  if ((await queueBtn.getAttribute('aria-label')) === 'Add to queue') {
    // The button flips optimistically (store state), so it does NOT prove the write landed.
    // Wait for the write to actually persist (2xx) before reading the queue view — otherwise,
    // under load, the GET can race ahead of the write ("empty").
    //
    // POST /queue/items, not PUT /queue (#1910/#1925): adding is an ITEM operation now. The
    // whole-list PUT is reserved for reordering, because replacing the list means a write made
    // offline and replayed later silently clobbers whatever another device did in between.
    const writePersisted = page.waitForResponse(
      (r) =>
        r.url().includes('/api/app/queue/items') && r.request().method() === 'POST' && r.ok(),
    )
    await queueBtn.click()
    await writePersisted
  }
  await expect(card.getByRole('button', { name: 'Remove from queue' })).toBeVisible()

  // The queue view (auth-gated route) lists the queued episode, served by the real API.
  await page.goto('/queue')
  await expect(page).toHaveURL(/\/queue/)
  await expect(page.getByText('Index Investing Without the Myths').first()).toBeVisible()
})
