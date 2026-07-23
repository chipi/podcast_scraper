import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * Search v3 §S6 (#1236) — "Search within this episode" rail launcher.
 *
 * When the Episode subject rail is open, EpisodeDetailPanel exposes a
 * "Search within episode" button that:
 *   1. Sets ``search.filters.episodeId`` to the exact episode_id.
 *   2. Clears sibling scope filters (feed / topic / speaker) so the wire
 *      matches the mental model of "this episode only".
 *   3. Switches ``mainTab`` to ``'search'`` and runs the query.
 *   4. Emits an ``episode_id=…`` param on ``/api/search`` for the server
 *      to scope the top-k retrieval (Search v3 §S6 server change).
 *
 * Also pins the ``SearchEpisodeChip`` on the filter bar: only visible
 * when the scope is active; clicking it clears the scope.
 */
test.describe('Search — rail launcher: search within this episode (#1236)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          search_api: true,
        }),
      })
    })
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
      })
    })
    await page.route('**/api/corpus/feeds**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          feeds: [{ feed_id: 'f1', display_title: 'Mock Show', episode_count: 1 }],
        }),
      })
    })
    await page.route('**/api/corpus/episodes**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          feed_id: null,
          items: [
            {
              metadata_relative_path: 'metadata/ep1.metadata.json',
              feed_id: 'f1',
              feed_display_title: 'Mock Show',
              topics: [],
              summary_title: 'A mock episode',
              summary_bullets_preview: [],
              summary_preview: 'A mock episode',
              episode_id: 'ep-ci-fixture',
              episode_title: 'Mock Episode Title',
              publish_date: '2026-04-15',
              has_gi: true,
              has_kg: true,
              cil_digest_topics: [],
            },
          ],
          next_cursor: null,
        }),
      })
    })
    // Silences the noisy loadFeedsAndIndex / similar sub-fetches so the
    // Episode rail lands in its populated state instead of retrying
    // sub-requests that would otherwise time out the mock stack.
    await page.route('**/api/index/stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          available: false,
          reason: 'mock-off',
          stats: null,
          reindex_recommended: false,
        }),
      })
    })
    await page.route('**/api/corpus/episodes/similar**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          source_metadata_relative_path: 'metadata/ep1.metadata.json',
          query_used: '',
          items: [],
          error: null,
          detail: null,
        }),
      })
    })
    await page.route('**/api/corpus/episodes/detail**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          metadata_relative_path: 'metadata/ep1.metadata.json',
          feed_id: 'f1',
          episode_id: 'ep-ci-fixture',
          episode_title: 'Mock Episode Title',
          publish_date: '2026-04-15',
          summary_title: 'A mock episode',
          summary_bullets: ['One bullet'],
          summary_text: null,
          has_gi: true,
          has_kg: true,
          cil_digest_topics: [],
        }),
      })
    })
    // /api/search records the URL so tests can assert the episode_id param.
    // Only returns hits when episode_id matches (proves the server-side scope
    // wire; the mock plays the role the real server plays).
    await page.route('**/api/search?**', async (route) => {
      const url = route.request().url()
      const params = new URL(url).searchParams
      const scoped = params.get('episode_id')
      const results =
        scoped === 'ep-ci-fixture'
          ? [
              {
                doc_id: 'insight:ep-ci-fixture:1',
                score: 0.9,
                source_tier: 'insight',
                text: 'An insight from the scoped episode.',
                metadata: { doc_type: 'insight', episode_id: 'ep-ci-fixture' },
              },
            ]
          : []
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ query: params.get('q') ?? '', results, query_type: 'semantic' }),
      })
    })
  })

  async function openEpisodeRail(page: import('@playwright/test').Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await expect(page.getByTestId('episode-detail-search-in-episode')).toBeVisible()
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
    await openEpisodeRail(page)
    const scopedRequest = page.waitForRequest((r) => {
      if (!r.url().includes('/api/search')) return false
      return /[?&]episode_id=ep-ci-fixture(?:&|$)/.test(r.url())
    })
    await page.getByTestId('episode-detail-search-in-episode').click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    // Episode scope chip appears with the id.
    const chip = page.getByTestId('search-chip-episode')
    await expect(chip).toBeVisible()
    await expect(chip).toContainText('Episode')
    // Search fired with episode_id param.
    await scopedRequest
    // Hit rendered (mock only returns hits when scope matches).
    await expect(page.getByTestId('search-workspace').locator('article').first()).toBeVisible()
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
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible()
    await expect(page.getByTestId('search-chip-episode')).toHaveCount(0)
  })
})
