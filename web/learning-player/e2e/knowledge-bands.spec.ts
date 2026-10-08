import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * The knowledge bands that had unit tests and no e2e (E2E_SURFACE_MAP coverage gaps, 2026-09-03).
 *
 * All of them render the knowledge layer where the listener already is, and all of them follow one
 * rule that only a browser can check: **absent intelligence omits cleanly**. A unit test feeds a
 * component props and sees it draw; it cannot see a band left half-rendered against a real corpus,
 * which is the failure these guard.
 */

test('the podcast signals band separates DISTINCTIVE topics from the rest', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'podcast-signals', testInfo)
  await page.goto('/podcast/p05')

  const band = page.getByTestId('podcast-signals')
  await expect(band).toBeVisible()

  // The split is the point (UXS-013): topics with `lift` above the corpus base rate are what set
  // this show apart, and they are listed under their own heading. Collapsing the two groups is how
  // a show's signature topic lost an alphabetical tiebreak to wallpaper every show covers.
  await expect(page.getByTestId('ps-distinctive-heading')).toBeVisible()
  await expect(page.getByTestId('ps-distinctive-topic').first()).toBeVisible()
  await expect(page.getByTestId('ps-topics-heading')).toBeVisible()
})

test('the show activity chart renders one bar per period', async ({ page }, testInfo) => {
  await signInIsolated(page, 'show-activity', testInfo)
  await page.goto('/podcast/p05')

  await expect(page.getByTestId('show-activity')).toBeVisible()
  // "Is this show alive?" is the question, so more than one bucket has to render for the shape to
  // mean anything.
  expect(await page.locator('[data-testid^="show-activity-bar-"]').count()).toBeGreaterThan(1)
})

test('tapping an activity bar jumps to that month in the episode list', async ({ page }, testInfo) => {
  await signInIsolated(page, 'show-activity-jump', testInfo)
  await page.goto('/podcast/p05')
  await expect(page.getByTestId('show-activity-unit')).toHaveText('Episodes per month')
  // The OLDEST month with episodes: furthest down the newest-first list, so the jump has to scroll.
  const bar = page.locator('button[data-testid^="show-activity-bar-"]').first()
  await expect(bar).toBeVisible()
  const month = ((await bar.getAttribute('data-testid')) ?? '').replace('show-activity-bar-', '')
  // The episode it should land on: the first (newest) one of that month in the page's own list.
  const list = (await (await page.request.get('/api/app/podcasts/p05/episodes?page=1&page_size=20')).json()) as {
    items: { slug: string; publish_date: string | null }[]
  }
  const target = list.items.find((e) => (e.publish_date ?? '').startsWith(month))
  expect(target, `no loaded episode in ${month}`).toBeTruthy()

  await bar.click()
  await expect(page.locator(`[data-episode-slug="${target!.slug}"]`)).toBeInViewport()
})

test('the player insight density band shows where the insights sit', async ({ page }, testInfo) => {
  await signInIsolated(page, 'insight-density', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  await expect(page.getByTestId('player-insight-density')).toBeVisible()
  // One band element per tick — `.first()` because the testid is the SERIES, not a singleton.
  await expect(page.getByTestId('player-density-band').first()).toBeVisible()
  await expect(page.getByTestId('player-density-tick').first()).toBeVisible()

  // Ticks mark WHERE the insights are; more than one is what makes the band informative rather
  // than decorative.
  expect(await page.getByTestId('player-density-tick').count()).toBeGreaterThan(1)

  // The episode notes no longer carry their own early/mid/late density box (operator 2026-10-08);
  // this band is the one place the app shows where the insights sit.
})

test('tapping the density band seeks there, like the scrubber (operator 2026-09-30)', async ({ page }, testInfo) => {
  await signInIsolated(page, 'insight-density-seek', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()

  // The scrubber carries the real position; wait for the audio's duration to reach it.
  const scrubber = page.locator('input[type="range"]').first()
  await expect.poll(async () => Number(await scrubber.getAttribute('max'))).toBeGreaterThan(0)
  const max = Number(await scrubber.getAttribute('max'))

  const band = page.getByTestId('player-density-seek')
  // A raw mouse click at page coordinates hits nothing off-screen, and on desktop the band sits
  // below the fold — bring it into view first, then measure.
  await band.scrollIntoViewIfNeeded()
  const box = (await band.boundingBox())!
  await page.mouse.click(box.x + box.width * 0.6, box.y + box.height / 2)

  // 60% along the band = 60% of the episode, within a second of rounding.
  await expect
    .poll(async () => Number(await scrubber.inputValue()))
    .toBeGreaterThanOrEqual(Math.floor(max * 0.6) - 1)
  expect(Number(await scrubber.inputValue())).toBeLessThanOrEqual(Math.ceil(max * 0.6) + 1)
})

test('the knowledge panel opens in place and closes with Escape', async ({ page }, testInfo) => {
  await signInIsolated(page, 'knowledge-panel', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()

  const open = page.getByTestId('player-open-insights')
  await expect(open).toBeVisible()
  await open.click()

  const panel = page.getByTestId('knowledge-panel')
  await expect(panel).toBeVisible()
  await expect(page.getByTestId('kp-insights')).toBeVisible()

  // UXS-014: on MOBILE the panel is a modal dialog and must be dismissible from the keyboard; on
  // desktop it is a side column that is always present and has nothing to dismiss. Asserting
  // Escape unconditionally would demand the desktop layout behave like the phone one.
  const isDialog = (await panel.getAttribute('role')) === 'dialog'
  if (isDialog) {
    await page.keyboard.press('Escape')
    await expect(panel).toBeHidden()
  } else {
    await expect(panel).toBeVisible()
  }
})

test('the activity chart labels its months and names each bar for a screen reader', async ({ page }, testInfo) => {
  await signInIsolated(page, 'show-activity-axis', testInfo)
  await page.goto('/podcast/p05')
  const axis = page.getByTestId('show-activity-axis')
  await expect(axis).toBeVisible()
  // The month under the bars, with a year at the first bar.
  await expect(axis).toContainText(/(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/)
  await expect(axis).toContainText(/20\d\d/)
  const bar = page.locator('button[data-testid^="show-activity-bar-"]').first()
  await expect(bar).toHaveAttribute('aria-label', /: \d+ episodes?$/)
})

test('the episode notes name every pill’s kind, and the THEME pill opens the theme on top', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'kp-mixed-kinds', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await page.getByTestId('player-open-insights').click()
  const panel = page.getByTestId('knowledge-panel')
  await expect(panel).toBeVisible()
  const chips = panel.locator('[data-testid="kp-topic-chip"], [data-testid="kp-person-chip"]')
  await expect(chips.first()).toBeVisible()
  const n = await chips.count()
  expect(n).toBeGreaterThan(1)
  // A MIXED group: every chip carries its kind label, not just some.
  await expect(panel.locator('[data-testid="kp-topic-chip"] [data-testid="kp-chip-kind"], [data-testid="kp-person-chip"] [data-testid="kp-chip-kind"]')).toHaveCount(n)
  await panel.getByTestId('kp-theme-link').click()
  await expect(page.getByTestId('theme-card')).toBeVisible()
})
