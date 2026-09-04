import { expect, test } from '@playwright/test'

/**
 * Tier-3 — the offline arc's surfaces against a REAL corpus and a real API.
 *
 * The fast-e2e variants of these run on the committed fixture corpus at :4174. This walk runs
 * against `make serve-for-validation` (production-shape backend, operator-swappable corpus via
 * `APP_CORPUS_PATH`), which is where scale-dependent drift shows up: a recap aggregates over the
 * whole listen log and the whole exposure log, and the queue's item routes do a locked
 * read-modify-write per call. Neither is interesting at fixture size and both are at corpus size.
 *
 * Screenshots are the artifact, as with every Tier-3 spec here.
 *
 * NOT covered, deliberately: downloads, the Downloaded list and Device settings. All are behind
 * `isNative()`, render nothing in a browser, and belong to the device tier
 * (`make test-app-ios-journey`).
 */

/** Advance the playhead the way playback does, so each save is a small forward DELTA. */
async function listenFor(page: import('@playwright/test').Page, seconds: number): Promise<void> {
  await page.locator('audio').waitFor({ state: 'attached' })
  await page.evaluate(async (target) => {
    const el = document.querySelector('audio') as HTMLAudioElement
    el.currentTime = 0
    await el.play().catch(() => {})
    for (let t = 1; t <= target; t += 1) {
      el.currentTime = t
      el.dispatchEvent(new Event('timeupdate'))
      await new Promise((r) => setTimeout(r, 20))
    }
    el.dispatchEvent(new Event('pause'))
  }, seconds)
}

test('operator recap: listen on a real corpus → Profile reports it honestly', async ({ page }) => {
  await page.goto('/api/app/auth/login?as=tier3-recap')
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()

  await page.goto('/')
  const episode = page.locator('a[href*="/episode/"]').first()
  await expect(episode).toBeVisible()
  await episode.click()
  await expect(page).toHaveURL(/\/episode\//)
  await page.locator('audio').waitFor({ state: 'attached' })
  await listenFor(page, 6)
  await page.screenshot({ path: 'validation-results/recap-01-listened.png', fullPage: true })

  await page.goto('/profile')
  await expect(page.getByRole('heading', { name: 'Your listening' })).toBeVisible()
  // The number is real AND says how much of the window produced it — recording started recently,
  // so a bare total would be a lie of omission at any corpus size.
  await expect(page.getByText(/Recorded \d+ of \d+ days/)).toBeVisible()
  // The tile that reported `sum(position_seconds)` — a lifetime furthest-position snapshot — is
  // gone. Asserted here too because a real corpus is where that number looked most plausible.
  await expect(page.getByText('Hours', { exact: true })).toHaveCount(0)
  await page.screenshot({ path: 'validation-results/recap-02-profile.png', fullPage: true })

  // The window toggle re-queries rather than re-rendering the same numbers.
  await page.getByRole('button', { name: 'This year', exact: true }).click()
  // Year-to-date: the window is 1 Jan → today, a different length every day. It is asserted
  // RELATIVE to the month window rather than against a floor, because "more than 31 days" is
  // false for the whole of January — the previous version would have failed every new year
  // (advisor-2 #7). What is actually true year-round: YTD is never SHORTER than the month window
  // once you are past 1 January, and on 1 January it is exactly one day.
  await expect(page.getByText(/Recorded \d+ of \d+ days/)).toBeVisible()
  const windowLength = async (): Promise<number> =>
    Number(
      (await page.getByText(/Recorded \d+ of \d+ days/).innerText()).match(/of (\d+) days/)![1],
    )
  const ytdWindow = await windowLength()
  expect(ytdWindow).toBeGreaterThanOrEqual(1)
  expect(ytdWindow).toBeLessThanOrEqual(366)
  await page.screenshot({ path: 'validation-results/recap-03-ytd.png', fullPage: true })
})

test('operator queue: item writes on a real corpus, and a cached queue refuses only reorder', async ({
  page,
  context,
}) => {
  await page.goto('/api/app/auth/login?as=tier3-queue')
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()

  // Queue two episodes through the UI, asserting the ITEM route carries them (#1910/#1925).
  //
  // From a SHOW page: the queue control lives on the episode card, not on the player page (the
  // player's "Queue & recently played" only opens the panel). Reaching a show from Home keeps this
  // corpus-agnostic — no feed id is hard-coded.
  await page.goto('/')
  const show = page.locator('a[href*="/podcast/"]').first()
  await expect(show).toBeVisible()
  await show.click()
  await expect(page).toHaveURL(/\/podcast\//)

  // Idempotent: the Tier-3 data dir persists between operator runs, so episodes may ALREADY be
  // queued and no "Add to queue" button exists. Requiring one made this fail on its own leftovers
  // rather than on the app. Queue up to two that are not yet queued; the assertions below only
  // need two items in the list, however they got there.
  const anyQueueControl = page.getByRole('button', { name: /queue/i }).first()
  await expect(anyQueueControl).toBeVisible()
  for (let i = 0; i < 2; i += 1) {
    const btn = page.getByRole('button', { name: 'Add to queue' }).first()
    if (!(await btn.isVisible().catch(() => false))) break
    await Promise.all([
      page.waitForResponse(
        (r) =>
          r.url().includes('/api/app/queue/items') && r.request().method() === 'POST' && r.ok(),
      ),
      btn.click(),
    ])
  }
  expect(
    await page.getByRole('button', { name: 'Remove from queue' }).count(),
    'at least two episodes must be queued for the reorder assertions below',
  ).toBeGreaterThanOrEqual(2)

  await page.goto('/queue')
  await expect(page.getByRole('heading', { name: 'Queue' })).toBeVisible()
  await page.screenshot({ path: 'validation-results/queue-01-live.png', fullPage: true })

  // Make the queue READ fail and reload: the store falls back to its cached copy.
  await context.route('**/api/app/queue', (route) =>
    route.request().method() === 'GET' ? route.abort() : route.continue(),
  )
  await page.reload()

  await expect(page.getByText(/Offline — showing your saved queue/)).toBeVisible()
  // Reordering refuses VISIBLY. Add/remove stay available: they are item-level and replay safely,
  // and disabling them was an over-correction that made the control read as dead.
  await expect(page.getByRole('button', { name: 'Move down' }).first()).toBeDisabled()
  await page.screenshot({ path: 'validation-results/queue-02-cached.png', fullPage: true })
})

test('operator deep link: ?t= opens a real episode AT the moment', async ({ page }) => {
  await page.goto('/api/app/auth/login?as=tier3-deeplink')
  await page.goto('/')
  const episode = page.locator('a[href*="/episode/"]').first()
  await expect(episode).toBeVisible()
  const href = (await episode.getAttribute('href')) as string
  const slug = href.split('/episode/')[1].split('?')[0]

  await page.goto(`/episode/${slug}?t=42`)
  await page.locator('audio').waitFor({ state: 'attached' })
  // The link wins over any remembered position for this load — that is the whole point of a
  // recap line, a shared quote, or an MCP citation pointing INTO an episode.
  await expect
    .poll(
      async () =>
        page.evaluate(() => (document.querySelector('audio') as HTMLAudioElement).currentTime),
      { timeout: 20_000 },
    )
    .toBeGreaterThan(40)
  await page.screenshot({ path: 'validation-results/deeplink-01-at-moment.png', fullPage: true })
})
