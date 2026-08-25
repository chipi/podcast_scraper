import { expect, test } from '@playwright/test'
import { navTo } from './helpers'

/**
 * Player-surface Queue & Recently-played panel (#1838).
 *
 * The Up-next / Recently-played lists used to be Library tabs; they now open FROM the player — a
 * queue button on the full player (next to the speed pill) and on the mini-player. This proves that
 * wiring at the level the component test can't: a real browser, the real modal, opened off the real
 * transport, and dismissed. Contents (queue rows, recent rows) are covered by QueuePanel.test.ts and
 * queue-reorder.spec — here we assert the surface exists where #1838 moved it.
 */

test('the full player opens the Queue & Recent panel and dismisses it', async ({ page }) => {
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

test('the mini-player opens the same Queue panel from anywhere', async ({ page }) => {
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

  await page.getByTestId('mini-player-queue').click()
  const panel = page.getByTestId('queue-panel')
  await expect(panel).toBeVisible()
  await expect(panel.getByRole('heading', { name: 'Up next' })).toBeVisible()

  await page.getByTestId('queue-panel-close').click()
  await expect(panel).toHaveCount(0)
})
