import { expect, test, type Page } from '@playwright/test'
import { expectSignedIn } from '../helpers'

/**
 * Shows and episodes on ONE page, where their controls sit (operator 2026-10-05).
 *
 * "When we have shows and episodes on the same surface I need the controls to look the same." The
 * three pages that put both on one screen, each built as a real state and shot:
 *
 *   1. Library › Saved — a saved show above a saved episode (show controls moved under the art).
 *   2. Search — a query that matches a show by title AND its episodes by content.
 *   3. Discover › Shows in LIST view — the Episodes tab next door puts controls under the art.
 *   4. Home › What's new — the same three actions on every position, stacked on 02+.
 *
 * Asserted before every capture, so an empty section FAILS instead of producing a PNG of nothing.
 *
 *   npm run design:shots -- show-episode-controls
 */
const VARIANT = process.env.DESIGN_VARIANT || 'baseline'
const shot = (name: string) =>
  `design-results/${VARIANT}/${test.info().project.name}/controls-${name}.png`

/** Fresh per run: this spec SAVES things, so a fixed identity would accumulate across runs. */
const IDENTITY = `design-controls-${Date.now().toString(36)}`

async function settle(page: Page): Promise<void> {
  await page.waitForLoadState('networkidle')
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => r(null))))
}

test('shows and episodes on one page share one place for controls', async ({ page }) => {
  await page.goto(`/api/app/auth/login?as=${IDENTITY}`)
  await expectSignedIn(page)

  // Save one show and one of its episodes, through the API the heart uses.
  const eps = await (await page.request.get('/api/app/podcasts/p01/episodes')).json()
  const slug = (eps as { items: { slug: string }[] }).items[0].slug
  for (const item of [
    { kind: 'show', ref: 'p01', label: 'Singletrack Sessions' },
    { kind: 'episode', ref: slug },
  ]) {
    const r = await page.request.put('/api/app/favorites', { data: item })
    expect(r.ok(), `PUT /favorites ${JSON.stringify(item)} → ${r.status()}`).toBe(true)
  }

  // 1. Library › Saved
  await page.goto('/library?tab=saved')
  await settle(page)
  const savedShows = page.getByTestId('saved-shows-list')
  await expect(savedShows.getByTestId('show-row-actions')).toBeVisible()
  await page.screenshot({ path: shot('library-saved'), fullPage: true })

  // 2. Search — a show title that the episodes' own words also carry.
  await page.goto('/search?q=Singletrack')
  await settle(page)
  await expect(page.getByTestId('search-shows')).toBeVisible({ timeout: 30_000 })
  await page.screenshot({ path: shot('search'), fullPage: true })
  // ...and the show's ⋯ open, so the menu's contents are on record too.
  await page.getByTestId('search-shows').getByTestId('show-row-menu').getByTestId('overflow-trigger').click()
  await expect(page.getByTestId('overflow-menu').getByTestId('follow-show')).toBeVisible()
  await page.screenshot({ path: shot('search-show-menu'), fullPage: false })
  await page.keyboard.press('Escape')

  // 4. Home › What's new — ♡ queue ⋯ on every position: a row on #01, a column on 02+.
  await page.goto('/')
  await settle(page)
  const actions = page.locator('[data-testid="episode-actions"].flex-col')
  await expect(actions.nth(1)).toBeVisible()
  // The #01 card at the TOP of the screen, so its row and the 02+ columns below are both in shot.
  await actions.first().evaluate((el) => el.closest('section')?.scrollIntoView({ block: 'start' }))
  await page.screenshot({ path: shot('home-whats-new'), fullPage: false })

  // 3. Discover › Shows, list view
  await page.goto('/browse?tab=shows')
  await settle(page)
  await page.getByTestId('show-view').click()
  await page.getByTestId('show-view-opt-list').click()
  await expect(page.getByTestId('show-browse-list').getByTestId('show-row-actions').first()).toBeVisible()
  await page.getByTestId('show-browse-list').scrollIntoViewIfNeeded()
  await page.screenshot({ path: shot('discover-shows-list'), fullPage: false })
})
