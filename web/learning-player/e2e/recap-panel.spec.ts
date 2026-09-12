import { expect, test, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Post-episode recap panel (RFC-122 / #2038) over a REAL API, NO mocks.
 *
 * When an episode finishes, the recap replaces the transport in place: what the listener just heard,
 * consolidated, with a "listen more like this" grid. Two flows:
 *   1. Empty queue — the recap appears and STOPS (dismiss returns to the finished player).
 *   2. A next episode queued — the recap becomes an end-card that counts down and continues the
 *      queue (reinforcement AND the queue's purpose), driven by the recap rather than by onEnded.
 *
 * "Finish" is dispatched directly on the <audio> element: real audio decode is not needed, and the
 * store's `ended` handler is what the whole flow hangs off. (headless Chromium's decoder is strict
 * with the fixture data-URL audio, so a synthetic `ended` is also the robust trigger.)
 */

/** Force the current episode to finish — the store's `ended` handler drives everything downstream. */
async function finishEpisode(page: Page): Promise<void> {
  await page.locator('[data-testid="app-audio"]').waitFor({ state: 'attached', timeout: 15_000 })
  await page.evaluate(() => document.querySelector('audio')?.dispatchEvent(new Event('ended')))
}

/** Add a specific episode card to the queue, idempotently (green on a re-run against a warm api). */
async function queueEpisode(page: Page, title: string): Promise<void> {
  const card = page.locator('article').filter({ hasText: title })
  const btn = card.getByRole('button', { name: /queue/i })
  await expect(btn).toBeVisible()
  if ((await btn.getAttribute('aria-label')) === 'Add to queue') {
    const persisted = page.waitForResponse(
      (r) => r.url().includes('/api/app/queue/items') && r.request().method() === 'POST' && r.ok(),
    )
    await btn.click()
    await persisted
  }
  await expect(card.getByRole('button', { name: 'Remove from queue' })).toBeVisible()
}

test('empty queue: an episode finishing shows the recap, and dismiss returns to the player', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'recap-empty', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  await finishEpisode(page)

  // The recap appears in place of the transport, leading with the finish + the reassurance.
  const panel = page.getByTestId('episode-recap-panel')
  await expect(panel).toBeVisible()
  await expect(panel.getByText('You just finished')).toBeVisible()
  await expect(panel.getByText('We took notes for you')).toBeVisible()

  // Nothing queued → no end-card countdown; the "more like this" grid is the manual next step.
  await expect(page.getByTestId('recap-countdown')).toHaveCount(0)

  // Dismiss returns to the finished player — the recap is gone and the transport is back.
  await page.getByTestId('recap-back').click()
  await expect(panel).toHaveCount(0)
  await expect(
    page
      .getByRole('button', { name: 'Play', exact: true })
      .or(page.getByText(/couldn.*load the audio|Audio isn|Audio.*unavailable/i)),
  ).toBeVisible()
})

test('queued next: finishing shows a countdown end-card that continues the queue', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'recap-queue', testInfo)

  // Seed a queue of two episodes from the same 4-episode dive show (p03), in order: the one we play
  // first, then a follow-up. Queue add-order is the queue order, so nextAfter(first) is the second.
  const queueHydrated = page
    .waitForResponse((r) => /\/api\/app\/queue(\?|$)/.test(r.url()) && r.request().method() === 'GET')
    .catch(() => null)
  await page.goto('/podcast/p03')
  await queueHydrated
  await queueEpisode(page, 'Plan the Dive, Manage the Risk')
  await queueEpisode(page, 'Wreck Diving Fundamentals')

  // Play the FIRST-queued episode, so the queue has a next after it.
  await page.getByText('Plan the Dive, Manage the Risk').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  await finishEpisode(page)

  // With a next queued, the recap is an end-card: a countdown + "Play next now".
  const panel = page.getByTestId('episode-recap-panel')
  await expect(panel).toBeVisible()
  await expect(page.getByTestId('recap-countdown')).toBeVisible()
  const playNext = page.getByTestId('recap-play-next')
  await expect(playNext).toBeVisible()

  // Continue the queue — the end-card advances and the recap closes.
  await playNext.click()
  await expect(panel).toHaveCount(0)
})
