import { expect, test, type Page } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * Moments (operator 2026-10-10): the obi's Moments door plays the episode's strongest moments back
 * to back, INSIDE the artwork — the page stays the episode page (masthead, artwork, transport). The
 * artwork shows the current moment and the reel's segments; the obi's top door reads Episode; the
 * transport's skips become previous / next moment; every moment is listed under the transport.
 * Keep listening here returns to the episode at that moment; Episode returns to where the listener
 * was. Step (‹ ›) moves between insights in the normal episode view.
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

  const card = page.getByTestId('moments-card')
  await expect(card).toBeVisible()
  await expect(page).toHaveURL(/moments=1/)
  // The episode page stays; the reel plays inside its artwork, listed under the transport.
  await expect(page.getByTestId('player-hero').getByTestId('moments-card')).toBeVisible()
  await expect(page.getByTestId('player-obi-episode')).toBeVisible()
  await expect(page.getByTestId('moments-index').getByRole('heading', { name: 'Moments', exact: true })).toBeVisible()
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
  await expect(page.getByTestId('moments-card')).toBeVisible()
  await expect.poll(() => audioTime(page)).toBeGreaterThanOrEqual(5.5)

  await page.getByTestId('moments-keep').click()
  await expect(page.getByTestId('moments-card')).toHaveCount(0)
  await expect(page.getByTestId('player-zone-d-live').or(page.getByTestId('player-zone-d-rest')).first()).toBeVisible()
  await expect(page).not.toHaveURL(/moments=1/)
  expect(await audioTime(page)).toBeGreaterThanOrEqual(5.5)
  expect(await audioPaused(page)).toBe(false)
})

test('the Episode door returns to where the listener was, paused', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'close')
  const before = await audioTime(page)
  await page.getByTestId('player-open-moments').click()
  await expect.poll(() => audioTime(page)).toBeGreaterThanOrEqual(5.5)

  await page.getByTestId('player-obi-episode').click()
  await expect(page.getByTestId('moments-card')).toHaveCount(0)
  await expect(page.getByTestId('player-open-moments')).toBeVisible()
  await expect.poll(() => audioTime(page)).toBeLessThan(before + 1)
  expect(await audioPaused(page)).toBe(true)
})

test('Step: › in the episode view jumps to the next insight', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'step')
  await page.getByTestId('player-step-next-rest').click()
  // The Bessent Tape's first insight is at 1:00 (2026-10-12): the host's closing "thanks for the
  // conversation" stopped being an insight and the episode's next claim, at 1:00, took its place.
  // (1:24 since ADR-165, 2026-10-11 — the 0:06 one was a guest's greeting.)
  // 1:24 again since 2026-10-13: the corpus rebuilt on main's claim ranking puts the episode's
  // first insight at 84 s (the GI's first SUPPORTED_BY quote).
  await expect.poll(() => audioTime(page)).toBeGreaterThanOrEqual(83.5)
  await expect.poll(() => audioTime(page)).toBeLessThan(90)
  // The jump is never unexplained: "› Jumped to 1:24" for a moment.
  await expect(page.getByTestId('player-step-flash').first()).toContainText(/Jumped to 1:2\d/)
})

test('Step: a left swipe on the insight card is next, a right swipe previous', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'swipe')
  const swipe = async (dx: number) => {
    const card = page.locator('[data-testid="player-zone-d-live"], [data-testid="player-zone-d-rest"]').first()
    // A swipe that lands on an insight swaps the rest card for the live one (a keyed transition);
    // measuring in between returned null under a loaded parallel run. Wait for a card with a box.
    let box: { x: number; y: number; width: number; height: number } | null = null
    await expect.poll(async () => (box = await card.boundingBox()) !== null).toBe(true)
    box = box!
    const x = box.x + box.width / 2
    const y = box.y + box.height / 2
    await card.dispatchEvent('pointerdown', { pointerType: 'touch', clientX: x, clientY: y, isPrimary: true })
    await card.dispatchEvent('pointerup', { pointerType: 'touch', clientX: x + dx, clientY: y + 4, isPrimary: true })
  }
  await swipe(-80)
  // The first insight, at 1:24 (see the step test above).
  await expect.poll(() => audioTime(page)).toBeGreaterThanOrEqual(83.5)
  const afterNext = await audioTime(page)
  await swipe(-80)
  await expect.poll(() => audioTime(page)).toBeGreaterThan(afterNext + 1)
  // The insight card is still on screen: the swipe did not count as the tap that hides it.
  await expect(page.getByTestId('player-zone-d-live')).toBeVisible()
  await swipe(80)
  await expect.poll(() => audioTime(page)).toBeLessThan(afterNext + 1)
})

test.describe('the ways into Moments', () => {
  test('the show page: no "▶ Moments" on the show line; the card opens the episode', async ({ page }, testInfo) => {
    // Operator 2026-10-14: "▶ Moments" competed with the show's name; it is the ⋯ item now.
    await signInIsolated(page, 'moments-entry-show', testInfo)
    await page.goto('/podcast/p05')
    const card = page.getByTestId('episode-card').filter({ hasText: 'The Bessent Tape' }).first()
    await expect(card).toBeVisible()
    await expect(card).not.toContainText('Moments')
    await card.getByRole('link', { name: /The Bessent Tape/ }).first().click()
    await expect(page).toHaveURL(/\/episode\/[^?]+$/)
  })

  test('any card: "Play moments" leads the ⋯ menu', async ({ page }, testInfo) => {
    await signInIsolated(page, 'moments-entry-menu', testInfo)
    await page.goto('/podcast/p05')
    const card = page.getByTestId('episode-card').filter({ hasText: 'The Bessent Tape' }).first()
    await card.getByTestId('overflow-trigger').click()
    const first = page.getByTestId('overflow-menu').getByRole('menuitem').first()
    await expect(first).toHaveAttribute('data-testid', 'episode-play-moments')
    await first.click()
    await expect(page.getByTestId('moments-card')).toBeVisible()
  })

  test('the Brief: "Play moments" right after the summary', async ({ page }, testInfo) => {
    await openEpisode(page, testInfo, 'entry-brief')
    await page.getByTestId('player-open-insights').click()
    // How many and how long, once the moments are loaded.
    await expect(page.getByTestId('kp-play-moments')).toContainText('Play 3 moments')
    await expect(page.getByTestId('kp-play-moments')).toContainText('min · the strongest moments, in order')
    await page.getByTestId('kp-play-moments').click()
    await expect(page.getByTestId('moments-card')).toBeVisible()
  })

  test('Discover: "Play moments" in a recommendation card ⋯ samples the episode', async ({ page }, testInfo) => {
    // Discover arc Step 3, "Sample an episode": every recommendation card offers the reel, through
    // the standard episode actions' ⋯ menu (operator 2026-10-11: the same actions as every card).
    await signInIsolated(page, 'moments-entry-discover', testInfo)
    await page.goto('/browse')
    const card = page.getByTestId('quote-cards').getByTestId('quote-card').first()
    await card.getByTestId('overflow-trigger').click()
    await page.getByTestId('episode-play-moments').click()
    await expect(page).toHaveURL(/moments=1/)
    await expect(page.getByTestId('moments-card')).toBeVisible()
  })

  test('search results: "Play moments" in an episode result\'s ⋯ menu', async ({ page }, testInfo) => {
    await signInIsolated(page, 'moments-entry-search', testInfo)
    await page.goto('/search?q=risk&scope=all')
    const group = page.getByTestId('episode-group').first()
    await expect(group).toBeVisible()
    await group.getByTestId('overflow-trigger').click()
    await page.getByTestId('episode-play-moments').click()
    await expect(page).toHaveURL(/moments=1/)
    await expect(page.getByTestId('moments-card')).toBeVisible()
  })
})

test.describe('Moments, everything around the reel', () => {
  test('the obi stays: Episode leads out, Brief and About open over the reel; the transport skips moments', async ({ page }, testInfo) => {
    await openEpisode(page, testInfo, 'doors')
    await page.getByTestId('player-open-moments').click()
    const obi = page.getByTestId('player-obi')
    await expect(obi.getByRole('button')).toHaveText(['Episode', 'Brief', 'About'])
    // The transport's skips are previous / next moment while the reel plays.
    await expect(page.getByTestId('moments-current')).toContainText('Moment 1 of 3')
    await page.getByTestId('player-skip-forward').click()
    await expect(page.getByTestId('moments-current')).toContainText('Moment 2 of 3')
    await page.getByTestId('player-skip-back').click()
    await expect(page.getByTestId('moments-current')).toContainText('Moment 1 of 3')
    await page.getByTestId('player-open-insights').click()
    await expect(page.getByTestId('kp-play-moments')).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('moments-card')).toBeVisible()
    await page.getByTestId('player-obi-episode').click()
    await expect(page.getByTestId('moments-card')).toHaveCount(0)
    await expect(page.getByTestId('player-skip-forward')).toHaveText('30↻')
  })

  test('every control has a spoken name (screen-reader pass)', async ({ page }, testInfo) => {
    await openEpisode(page, testInfo, 'a11y')
    await expect(page.getByRole('navigation', { name: 'Moments, brief and description' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Next insight' }).first()).toBeVisible()
    await expect(page.getByRole('button', { name: 'Previous insight' }).first()).toBeVisible()
    await page.getByTestId('player-open-moments').click()
    const card = page.getByTestId('moments-card')
    for (const name of ['Previous moment', 'Next moment']) {
      await expect(card.getByRole('button', { name })).toBeVisible()
      await expect(page.getByTestId('player-transport').getByRole('button', { name })).toBeVisible()
    }
    await expect(page.getByRole('button', { name: 'Episode', exact: true })).toBeVisible()
    await expect(page.getByTestId('player-transport').getByRole('button', { name: /^(Play|Pause)$/ })).toBeVisible()
    await expect(card.getByRole('button', { name: /Keep listening here/ })).toBeVisible()
    // The list under the transport says which moment is playing.
    await expect(page.getByTestId('moments-index').locator('[aria-current="true"]')).toHaveCount(1)
  })

  test('leaving the page mid-reel: the mini-player says "Moments · n / total" and brings you back', async ({ page }, testInfo) => {
    await openEpisode(page, testInfo, 'mini')
    await page.getByTestId('player-open-moments').click()
    await expect(page.getByTestId('moments-card')).toBeVisible()
    // Back to the show page in-app (the reel keeps playing), where the mini-player shows.
    await page.goBack()
    await expect(page.getByTestId('mini-player-moments')).toHaveText(/Moments · \d \/ 3/)
    await page.getByTestId('mini-player-open').click()
    await expect(page.getByTestId('moments-card')).toBeVisible()
  })

  test('the end card chains to the next queued episode\'s moments', async ({ page }, testInfo) => {
    await signInIsolated(page, 'moments-chain', testInfo)
    await page.goto('/podcast/p05')
    for (const title of ['The Bessent Tape', 'The Risk Panel: Diversify or Concentrate?']) {
      const q = page.locator('article').filter({ hasText: title }).first().getByRole('button', { name: 'Add to queue' })
      await expect(q).toBeVisible()
      await q.click()
    }
    await page.locator('article').filter({ hasText: 'The Bessent Tape' }).first().getByTestId('overflow-trigger').click()
    await page.getByTestId('episode-play-moments').click()
    await expect(page.getByTestId('moments-card')).toBeVisible()
    await page.getByTestId('moments-index-item').last().click()
    await page.getByTestId('moments-next').click()
    const next = page.getByTestId('moments-next-episode')
    await expect(next).toContainText('The Risk Panel')
    await next.click()
    await expect(page).toHaveURL(/moments=1/)
    await expect(page.getByRole('heading', { level: 1 })).toContainText('The Risk Panel')
  })

  test('offline: "Play moments" greys for an episode that is not downloaded, saying why', async ({ page, context }, testInfo) => {
    await signInIsolated(page, 'moments-offline', testInfo)
    await page.goto('/podcast/p05')
    const card = page.getByTestId('episode-card').filter({ hasText: 'The Bessent Tape' }).first()
    await expect(card).toBeVisible()
    // The network drops while the page is open (the browser's offline event): no downloads on the
    // web, so the ⋯ item greys in place, inert and saying why.
    await context.setOffline(true)
    await card.getByTestId('overflow-trigger').click()
    const item = page.getByTestId('episode-play-moments')
    await expect(item).toBeDisabled()
    await expect(item).toContainText('need a connection')
    await context.setOffline(false)
  })

  test('a topic page\'s episode rows open the episode (no "▶ Moments" beside them)', async ({ page }, testInfo) => {
    await signInIsolated(page, 'moments-topic', testInfo)
    await page.goto('/topic/' + encodeURIComponent('topic:risk-management'))
    await expect(page.getByTestId('topic-view')).toBeVisible()
    const row = page.getByTestId('episode-row').first()
    await expect(row).toBeVisible()
    await expect(row).not.toContainText('Moments')
    await row.getByRole('link').first().click()
    await expect(page).toHaveURL(/\/episode\/[^?]+$/)
  })
})
