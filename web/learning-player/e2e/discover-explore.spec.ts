import { expect, test } from '@playwright/test'
import { signInIsolated, showEveryonesTrends } from './helpers'

/**
 * Discover is "what to hear next"; Explore is for looking around (ADR-163 §1, operator 2026-10-11).
 * Real API over the committed corpus.
 *
 * - Discover leads with the quote cards: each says WHY it is here (`RecommendationWhy`) in the
 *   transcript's own words, and "Play from" opens the player at the quote.
 * - Trends and the Episodes · Shows catalog are on Explore, one tap from Discover's header; old
 *   `/browse?trends=` / `?tab=` / `#catalog` links land there.
 */

test('Discover leads with quote cards that say why, and play from the quote', async ({ page }, testInfo) => {
  await signInIsolated(page, 'discover-quotes', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  const cards = page.getByTestId('quote-card')
  await expect(cards.first()).toBeVisible()
  const first = cards.first()
  await expect(first.getByTestId('why-reason')).not.toBeEmpty()
  await expect(first.getByTestId('why-quote')).toContainText('“')
  // Episode-first (ADR-163 §2): the quote cards sit above trending shows.
  const quotesTop = (await page.getByTestId('quote-cards').boundingBox())!.y
  const showsTop = (await page.getByTestId('trending-shows-rail').boundingBox())!.y
  expect(quotesTop).toBeLessThan(showsTop)

  const play = first.getByTestId('why-play-from')
  await expect(play).toHaveAttribute('href', /\/episode\/[^?]+\?t=\d+&play=1$/)
  await play.click()
  await expect(page).toHaveURL(/\/episode\/[^?]+\?t=\d+/)
})

test('Explore holds Trends and the catalog, one tap from Discover', async ({ page }, testInfo) => {
  await signInIsolated(page, 'explore-home', testInfo)
  await page.goto('/browse')
  await expect(page.getByTestId('browse-tab-episodes')).toHaveCount(0)
  await page.getByTestId('discover-explore-link').click()
  await expect(page).toHaveURL(/\/explore$/)
  await expect(page.getByTestId('explore-view')).toBeVisible()
  await expect(page.getByTestId('discovery-explorer')).toBeVisible()
  await expect(page.getByTestId('browse-tab-episodes')).toHaveAttribute('aria-selected', 'true')
  await page.getByTestId('browse-tab-shows').click()
  await expect(page.getByTestId('browse-panel-shows')).toBeVisible()
})

test('old Discover links into Trends and the catalog land on Explore', async ({ page }, testInfo) => {
  await signInIsolated(page, 'explore-redirects', testInfo)
  await page.goto('/browse?trends=person')
  await expect(page).toHaveURL(/\/explore\?trends=person/)
  await expect(page.getByTestId('discovery-tab-person')).toHaveAttribute('aria-selected', 'true')
  await page.goto('/browse?tab=shows#catalog')
  await expect(page).toHaveURL(/\/explore\?tab=shows#catalog/)
  await expect(page.getByTestId('browse-tab-shows')).toHaveAttribute('aria-selected', 'true')
})

test("trending shows' See all opens Explore's Shows, sorted by trending", async ({ page }, testInfo) => {
  await signInIsolated(page, 'explore-see-all', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  await page.getByTestId('trending-shows-seeall').click()
  await expect(page).toHaveURL(/\/explore\?tab=shows&sort=trending&view=list#catalog/)
  await expect(page.getByTestId('browse-tab-shows')).toHaveAttribute('aria-selected', 'true')
})
