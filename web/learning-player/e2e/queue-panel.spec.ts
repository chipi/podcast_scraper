import { expect, test } from '@playwright/test'
import { navTo, signInIsolated } from './helpers'

/**
 * Player-surface Queue & Recently-played panel (#1838).
 *
 * The Up-next / Recently-played lists used to be Library tabs; they now open FROM the player — a
 * queue button on the full player (next to the speed pill) and on the mini-player. This proves that
 * wiring at the level the component test can't: a real browser, the real modal, opened off the real
 * transport, and dismissed. Contents (queue rows, recent rows) are covered by QueuePanel.test.ts and
 * queue-reorder.spec — here we assert the surface exists where #1838 moved it.
 */

test('the full player opens the Queue & Recent panel and dismisses it', async ({ page }, testInfo) => {
  await signInIsolated(page, 'queue-panel-full', testInfo)
  // Reach the episode via its show page — date-independent, same route the other specs use.
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  // The queue button is a static transport affordance (no playback required to open it).
  const openQueue = page.getByTestId('player-queue')
  await expect(openQueue).toBeVisible()
  await openQueue.click()

  const panel = page.getByTestId('queue-panel')
  await expect(panel).toBeVisible()
  await expect(panel.getByRole('heading', { name: 'Up next' })).toBeVisible()

  // Dismiss via the close button (ESC / backdrop paths are covered by QueuePanel.test.ts).
  await page.getByTestId('queue-panel-close').click()
  await expect(panel).toHaveCount(0)
})

test('the mini-player hands the queue to the masthead, which lands on the full page', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'queue-panel-mini', testInfo)
  // Start playback so the mini-player is present, then leave the player in-app.
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await page.getByRole('button', { name: 'Play', exact: true }).first().click()
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.currentTime ?? 0), {
      timeout: 15_000,
    })
    .toBeGreaterThan(0.2)

  await navTo(page, 'search')
  await expect(page.getByTestId('mini-player')).toBeVisible()

  /*
   * The mini-player's queue button is GONE (operator 2026-09-23). The masthead now carries a queue
   * control at every width, so the bar kept a second route to the same place on a screen that
   * already showed the first — and the bar is the most space-constrained strip in the app.
   *
   * Asserted as an ABSENCE as well as a replacement: a spec that only checks the new path would
   * stay green if the old button quietly came back.
   */
  await expect(page.getByTestId('mini-player-queue')).toHaveCount(0)
  // What the bar carries instead: save and collect, for the episode that is playing.
  await expect(page.getByTestId('mini-player').getByRole('button', { name: /save|remove/i }).first()).toBeVisible()

  await page.getByTestId('masthead-queue').click()
  await expect(page).toHaveURL(/\/queue/)
  // The destination shows BOTH halves — Up next and the history — because it is now the queue's
  // primary surface, not a subset of the panel reached from the player.
  await expect(page.getByRole('heading', { name: 'Up next' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Recently played' })).toBeVisible()
  await expect(page.getByTestId('queue-panel-recent')).toBeVisible()
})

test('the panel is still reachable from the FULL player, with both sections', async ({
  page,
}, testInfo) => {
  // The panel did not go away — only the mini-player's route into it. The full player keeps it,
  // because there the queue is a transport concern and a modal beats leaving the episode.
  await signInIsolated(page, 'queue-panel-player', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  await page.getByTestId('player-queue').click()
  const panel = page.getByTestId('queue-panel')
  await expect(panel).toBeVisible()
  await expect(panel.getByRole('heading', { name: 'Up next' })).toBeVisible()

  await page.getByTestId('queue-panel-close').click()
  await expect(panel).toHaveCount(0)
})
