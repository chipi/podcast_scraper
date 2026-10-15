import { expect, test, type Locator, type Page } from '@playwright/test'
import { closeListSearch, listSearch, signInIsolated } from './helpers'

/**
 * Explore's Browse — every control of both lists, ON Explore (operator 2026-10-11).
 *
 * For a day on this branch Explore's Episodes and Shows tabs showed a "taste with See all" instead
 * of the lists, and the filter bar — text filter, which episodes, state, sort, category, language,
 * list ⇄ grid — vanished from Explore. The lists' own tests opened them on their own pages, so
 * nothing noticed. These drive each control where the listener uses it: inside Explore's tabs.
 * REAL API over the committed corpus, no mocks.
 */

async function pick(page: Page, scope: Locator, testid: string, value: string): Promise<void> {
  await scope.getByTestId(testid).click()
  await page.getByTestId(`${testid}-opt-${value}`).click()
}

async function titles(rows: Locator): Promise<string[]> {
  return (await rows.allInnerTexts()).map((t) => t.trim()).filter(Boolean)
}

test('Episodes: text filter, which episodes, state, sort and list ⇄ grid all work on Explore', async ({ page }, testInfo) => {
  await signInIsolated(page, 'explore-episode-controls', testInfo)
  await page.goto('/explore')
  const panel = page.getByTestId('browse-panel-episodes')
  const cards = panel.getByTestId('episode-card')
  await expect(cards.first()).toBeVisible()
  const all = await cards.count()

  // Text filter narrows the list to what matches, and clearing it brings the list back.
  const search = await listSearch(panel)
  await search.fill('Bessent')
  await expect(cards.first()).toContainText('Bessent')
  await expect.poll(() => cards.count()).toBeLessThan(all)
  await search.fill('')
  await closeListSearch(panel)
  // Back to the whole list: at least the first page it showed before (a search loads the rest).
  await expect.poll(() => cards.count()).toBeGreaterThanOrEqual(all)

  // Which episodes: Following on a fresh account (it follows nothing) empties the list; All restores it.
  await pick(page, panel, 'catalog-from', 'following')
  await expect(panel.getByTestId('catalog-from')).toHaveText('Following')
  await expect(cards).toHaveCount(0)
  await pick(page, panel, 'catalog-from', 'all')
  await expect.poll(() => cards.count()).toBeGreaterThanOrEqual(all)

  // State: Played on a fresh account has nothing; All restores the list.
  await pick(page, panel, 'list-toolbar-filter', 'played')
  await expect(cards).toHaveCount(0)
  await pick(page, panel, 'list-toolbar-filter', 'all')
  await expect.poll(() => cards.count()).toBeGreaterThanOrEqual(all)

  // Sort: A–Z then Z–A reverse the order.
  // The card's title is its link to the player.
  const heading = cards.first().locator('a[href*="/episode/"]').first()
  await pick(page, panel, 'list-toolbar-sort', 'az')
  const firstAz = (await heading.innerText()).trim()
  await pick(page, panel, 'list-toolbar-sort', 'za')
  await expect(heading).not.toHaveText(firstAz)
  await pick(page, panel, 'list-toolbar-sort', 'newest')

  // List ⇄ grid.
  await pick(page, panel, 'list-toolbar-view', 'grid')
  await expect(panel.getByTestId('episode-grid')).toBeVisible()
  await expect(cards).toHaveCount(0)
  await pick(page, panel, 'list-toolbar-view', 'list')
  await expect(panel.getByTestId('episode-grid')).toHaveCount(0)
  await expect(cards.first()).toBeVisible()
})

test('Shows: text filter, sort, language and list ⇄ grid all work on Explore', async ({ page }, testInfo) => {
  await signInIsolated(page, 'explore-show-controls', testInfo)
  await page.goto('/explore')
  await page.getByTestId('browse-tab-shows').click()
  const panel = page.getByTestId('browse-panel-shows')
  const grid = panel.getByTestId('show-browse-grid')
  await expect(grid).toBeVisible()
  const tiles = grid.getByRole('listitem')
  const all = await tiles.count()
  expect(all).toBeGreaterThan(3)

  // Text filter.
  const showSearch = await listSearch(panel, 'show-browse-search')
  await showSearch.fill('Drift')
  await expect(tiles).toHaveCount(1)
  await expect(tiles.first()).toContainText('The Drift')
  await showSearch.fill('')
  await closeListSearch(panel, 'show-browse-search')
  await expect.poll(() => tiles.count()).toBe(all)

  // Sort: A–Z then Z–A reverse the order.
  await pick(page, panel, 'show-browse-sort', 'az')
  const firstAz = (await titles(tiles))[0]
  await pick(page, panel, 'show-browse-sort', 'za')
  await expect.poll(async () => (await titles(tiles))[0]).not.toBe(firstAz)

  // Language: the corpus spans several, so the chip is offered; one language narrows the list.
  await expect(panel.getByTestId('show-browse-language')).toBeVisible()
  await pick(page, panel, 'show-browse-language', 'es')
  await expect.poll(() => tiles.count()).toBeLessThan(all)
  await expect(tiles.first()).toBeVisible()
  await pick(page, panel, 'show-browse-language', '')
  await expect.poll(() => tiles.count()).toBe(all)

  // List ⇄ grid.
  await pick(page, panel, 'show-view', 'list')
  await expect(panel.getByTestId('show-browse-list')).toBeVisible()
  await expect(grid).toHaveCount(0)
  await pick(page, panel, 'show-view', 'grid')
  await expect(grid).toBeVisible()
})
