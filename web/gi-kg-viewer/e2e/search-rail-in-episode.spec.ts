import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  liveFirstEpisode,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
  type LiveEpisode,
} from './helpers'

/**
 * Search v3 §S6 (#1236) — "Search within this episode" rail launcher.
 *
 * When the Episode subject rail is open, EpisodeDetailPanel exposes a "Search within episode"
 * button that:
 *   1. Sets ``search.filters.episodeId`` to the exact episode_id.
 *   2. Clears sibling scope filters (feed / topic / speaker) so the wire matches the mental model
 *      of "this episode only".
 *   3. Switches ``mainTab`` to ``'search'`` and runs the query.
 *   4. Emits an ``episode_id=…`` param on ``/api/search`` for the server to scope the top-k
 *      retrieval (Search v3 §S6 server change).
 *
 * Also pins the ``SearchEpisodeChip`` on the filter bar: only visible when the scope is active;
 * clicking it clears the scope.
 *
 * #1619 — fully migrated to the live index, and this file gained real coverage in the process.
 *
 * The old version's `/api/search` stub returned hits **only** when `episode_id` matched, then
 * asserted hits rendered — so "the server scoped the retrieval" was a property of the stub, not of
 * the server. The live server really does scope: with `episode_id` set, all 8 returned hits carry
 * that episode_id, which this now asserts directly.
 */
test.describe('Search — rail launcher: search within this episode (#1236)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  /** Open the Library, focus the first live episode, and wait for the rail launcher. */
  async function openEpisodeRail(page: Page): Promise<LiveEpisode> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    const ep = await liveFirstEpisode(page)
    await page
      .getByRole('button', { name: `${ep.episode_title}, ${ep.feed_display_title}` })
      .click()
    await expect(page.getByTestId('episode-detail-search-in-episode')).toBeVisible()
    return ep
  }

  test('rail exposes "Search within episode" button (enabled when episode_id resolves)', async ({
    page,
  }) => {
    await openEpisodeRail(page)
    await expect(page.getByTestId('episode-detail-search-in-episode')).toBeEnabled()
  })

  test('click → switch to Search tab, filter chip visible, request URL carries episode_id', async ({
    page,
  }) => {
    const ep = await openEpisodeRail(page)
    const scopedRequest = page.waitForRequest((r) => {
      if (!r.url().includes('/api/search')) return false
      return new URL(r.url()).searchParams.get('episode_id') === ep.episode_id
    })
    await page.getByTestId('episode-detail-search-in-episode').click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    // Episode scope chip appears with the id.
    const chip = page.getByTestId('search-chip-episode')
    await expect(chip).toBeVisible()
    await expect(chip).toContainText('Episode')
    // Search fired with the episode_id param.
    await scopedRequest
    await expect(
      page.getByTestId('search-workspace').locator('article').first(),
    ).toBeVisible({ timeout: 30_000 })

    /* The point of §S6 is that the SERVER narrows retrieval, not that the UI drew a chip. Ask the
     * scoped endpoint directly and require every hit to belong to this episode — the old stub
     * simply refused to return anything unscoped, which proved nothing about the server. */
    const resp = await page.request.get(
      `/api/search?q=${encodeURIComponent('systems thinking')}&episode_id=${encodeURIComponent(ep.episode_id)}&top_k=8`,
    )
    const { results } = (await resp.json()) as {
      results: { metadata?: { episode_id?: string } }[]
    }
    expect(results.length).toBeGreaterThan(0)
    const foreign = results.filter((r) => r.metadata?.episode_id !== ep.episode_id)
    expect(foreign).toHaveLength(0)
  })

  test('clicking the SearchEpisodeChip clears the scope and hides the chip', async ({ page }) => {
    await openEpisodeRail(page)
    await page.getByTestId('episode-detail-search-in-episode').click()
    await expect(page.getByTestId('search-chip-episode')).toBeVisible()
    await page.getByTestId('search-chip-episode').click()
    await expect(page.getByTestId('search-chip-episode')).toHaveCount(0)
  })

  test('SearchEpisodeChip is NOT visible when no episode scope is active', async ({ page }) => {
    // Straight to Search — no rail launcher fired.
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible()
    await expect(page.getByTestId('search-chip-episode')).toHaveCount(0)
  })
})
