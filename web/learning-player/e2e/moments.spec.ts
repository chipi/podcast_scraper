import { expect, test, type Page } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * Moments (operator 2026-10-10): the obi's Moments door turns the player page into the Moments
 * view — its own title, the current moment leading, an index of all of them — and the reel plays
 * the moments back to back. Keep listening here returns to the episode at that moment; ✕ returns
 * to where the listener was. Step (‹ ›) moves between insights in the normal episode view.
 *
 * "The Bessent Tape" has three insights; the e2e stack sets APP_MOMENTS_CONFIG min_gap_seconds=5
 * (playwright.config.ts) because the fixture's quotes are synthetic and bunched in the first 48 s.
 */
async function openEpisode(page: Page, testInfo: import('@playwright/test').TestInfo, tag: string) {
  await signInIsolated(page, `moments-${tag}`, testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('The Bessent Tape').first().click()
  await expect(page.getByTestId('player-hero')).toBeVisible()
  await expect(page.getByTestId('player-open-moments')).toBeVisible()
}

const audioTime = (page: Page) =>
  page.evaluate(() => (document.querySelector('audio[data-testid="app-audio"]') as HTMLAudioElement).currentTime)
const audioPaused = (page: Page) =>
  page.evaluate(() => (document.querySelector('audio[data-testid="app-audio"]') as HTMLAudioElement).paused)

test('the Moments door opens the Moments view and the reel plays its moments in order', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'reel')
  await page.getByTestId('player-open-moments').click()

  const reel = page.getByTestId('moments-reel')
  await expect(reel).toBeVisible()
  await expect(page).toHaveURL(/moments=1/)
  // A different place, not the episode view with fewer controls.
  await expect(reel.getByRole('heading', { name: 'Moments', exact: true })).toBeVisible()
  await expect(page.getByTestId('player-hero')).toHaveCount(0)
  await expect(page.getByTestId('moments-segment')).toHaveCount(3)
  await expect(page.getByTestId('moments-current')).toContainText('Moment 1 of 3')
  await expect(page.getByTestId('moments-index-item')).toHaveCount(3)
  await expect.poll(() => audioTime(page)).toBeGreaterThanOrEqual(5.5)

  await page.getByTestId('moments-next').click()
  await expect(page.getByTestId('moments-current')).toContainText('Moment 2 of 3')
  await expect.poll(() => audioTime(page)).toBeGreaterThanOrEqual(11.5)

  // Any moment is a tap away in the index.
  await page.getByTestId('moments-index-item').nth(2).click()
  await expect(page.getByTestId('moments-current')).toContainText('Moment 3 of 3')
})

test('Keep listening here returns to the episode at that moment, still playing', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'keep')
  await page.getByTestId('player-open-moments').click()
  await expect(page.getByTestId('moments-reel')).toBeVisible()
  await expect.poll(() => audioTime(page)).toBeGreaterThanOrEqual(5.5)

  await page.getByTestId('moments-keep').click()
  await expect(page.getByTestId('moments-reel')).toHaveCount(0)
  await expect(page.getByTestId('player-hero')).toBeVisible()
  await expect(page).not.toHaveURL(/moments=1/)
  expect(await audioTime(page)).toBeGreaterThanOrEqual(5.5)
  expect(await audioPaused(page)).toBe(false)
})

test('✕ returns to where the listener was, paused', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'close')
  const before = await audioTime(page)
  await page.getByTestId('player-open-moments').click()
  await expect.poll(() => audioTime(page)).toBeGreaterThanOrEqual(5.5)

  await page.getByTestId('moments-close').click()
  await expect(page.getByTestId('player-hero')).toBeVisible()
  await expect.poll(() => audioTime(page)).toBeLessThan(before + 1)
  expect(await audioPaused(page)).toBe(true)
})

test('Step: › in the episode view jumps to the next insight', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'step')
  await page.getByTestId('player-step-next-rest').click()
  await expect.poll(() => audioTime(page)).toBeGreaterThanOrEqual(5.5)
  await expect.poll(() => audioTime(page)).toBeLessThan(12)
})

test.describe('the ways into Moments', () => {
  test('the show page: "▶ Moments" on the card\'s show line', async ({ page }, testInfo) => {
    await signInIsolated(page, 'moments-entry-show', testInfo)
    await page.goto('/podcast/p05')
    const card = page.getByTestId('episode-card').filter({ hasText: 'The Bessent Tape' }).first()
    await card.getByTestId('moments-link').click()
    await expect(page).toHaveURL(/moments=1/)
    await expect(page.getByTestId('moments-reel')).toBeVisible()
  })

  test('any card: "Play moments" leads the ⋯ menu', async ({ page }, testInfo) => {
    await signInIsolated(page, 'moments-entry-menu', testInfo)
    await page.goto('/podcast/p05')
    const card = page.getByTestId('episode-card').filter({ hasText: 'The Bessent Tape' }).first()
    await card.getByTestId('overflow-trigger').click()
    const first = page.getByTestId('overflow-menu').getByRole('menuitem').first()
    await expect(first).toHaveAttribute('data-testid', 'episode-play-moments')
    await first.click()
    await expect(page.getByTestId('moments-reel')).toBeVisible()
  })

  test('the Brief: "Play moments" above the summary', async ({ page }, testInfo) => {
    await openEpisode(page, testInfo, 'entry-brief')
    await page.getByTestId('player-open-insights').click()
    await page.getByTestId('kp-play-moments').click()
    await expect(page.getByTestId('moments-reel')).toBeVisible()
  })

  test('search results: "▶ Moments" on an episode result', async ({ page }, testInfo) => {
    await signInIsolated(page, 'moments-entry-search', testInfo)
    await page.goto('/search?q=risk&scope=all')
    const link = page.getByTestId('moments-link').first()
    await expect(link).toBeVisible()
    await link.click()
    await expect(page).toHaveURL(/moments=1/)
    await expect(page.getByTestId('moments-reel')).toBeVisible()
  })
})
