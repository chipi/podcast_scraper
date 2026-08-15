import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  liveFeeds,
  liveFirstEpisode,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * #1619 — migrated to the live API, except the four tests annotated below.
 *
 * `/api/corpus/{feeds,episodes,episodes/detail}` and `/api/relational/episode-insights` all serve
 * real data from the v3 fixture corpus, so the subjects here are read from the server rather than
 * hand-authored. What stays mocked is a **corpus state** (16 feeds, an episode with no peers) or
 * deliberate **fault injection** (`no_index`) — not an assertion that was too much work to rewrite.
 */

/**
 * #669 — the legacy "Filters" disclosure was replaced by the always-visible
 * LibraryFilterBar chip row. Tests that need to interact with the feed list
 * must open the LibraryFeedChip popover; tests that only touch episode rows
 * skip this helper.
 */
async function openLibraryFeedChip(page: Page): Promise<void> {
  const chip = page.getByTestId('library-chip-feed')
  if ((await chip.getAttribute('aria-expanded')) === 'false') {
    await chip.click()
  }
}

/** Open Library against the live corpus and wait for the root to render. */
async function openLibrary(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
  await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
  await expect(page.getByTestId('library-root')).toBeVisible()
}

/** The accessible name of a Library episode row: `"<title>, <feed>"`. */
function episodeRowName(ep: { episode_title: string; feed_display_title: string }): string {
  return `${ep.episode_title}, ${ep.feed_display_title}`
}

test.describe('Corpus Library tab', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('Episode subject rail: slash opens the command palette; episode rail stays visible (Search v3 §S3 + §S4-shell)', async ({
    page,
  }) => {
    await openLibrary(page)
    const ep = await liveFirstEpisode(page)
    await page.getByRole('button', { name: episodeRowName(ep) }).click()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: ep.episode_title }),
    ).toBeVisible()
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('/')
    // §S3 palette + §S4-shell: `/` summons the shell palette (there's no
    // launcher to focus anymore). The episode subject rail is a separate
    // right-rail surface and must stay visible behind the palette overlay.
    await expect(page.getByTestId('command-palette')).toBeVisible()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: ep.episode_title }),
    ).toBeVisible()
  })

  test('feed filter: Clear feed filter resets to all feeds', async ({ page }) => {
    await openLibrary(page)
    const feed = (await liveFeeds(page))[0]!
    // #669 — Clear lives inside the LibraryFeedChip popover and only renders
    // when a feed is selected (no disabled state in chip popover).
    await openLibraryFeedChip(page)
    const popover = page.getByTestId('library-popover-feed')
    await expect(popover.getByTestId('corpus-feed-filter-clear')).toHaveCount(0)
    await page
      .getByRole('button', {
        name: `${feed.display_title}, feed id ${feed.feed_id}, ${feed.episode_count} episodes`,
      })
      .click()
    await openLibraryFeedChip(page)
    const clearFeed = popover.getByTestId('corpus-feed-filter-clear')
    await expect(clearFeed).toBeVisible()
    await clearFeed.click()
    await openLibraryFeedChip(page)
    await expect(popover.getByTestId('corpus-feed-filter-clear')).toHaveCount(0)
  })

  test('lists feeds and episodes; search handoff fills query and feed filter', async ({
    page,
  }) => {
    await openLibrary(page)
    const ep = await liveFirstEpisode(page)
    const feed = (await liveFeeds(page)).find((f) => f.feed_id === ep.feed_id)!
    await openLibraryFeedChip(page)
    await expect(
      page.getByRole('button', {
        name: `${feed.display_title}, feed id ${feed.feed_id}, ${feed.episode_count} episodes`,
      }),
    ).toBeVisible()
    // Close the popover before clicking an episode behind it.
    await page.getByTestId('library-chip-feed').click()
    const row = page.getByRole('button', { name: episodeRowName(ep) })
    await expect(row).toBeVisible()
    await row.click()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: 'Key points' }),
    ).toBeVisible()
    await page.getByRole('button', { name: 'Prefill semantic search' }).click()
    /* The prefill is the server's own similarity query (title + bullets, not the prose summary —
     * see build_similarity_query). Assert it carries the episode's summary title rather than
     * re-deriving the exact concatenation, which would just restate the implementation. */
    await expect(page.locator('#search-q')).not.toHaveValue('')
    if (ep.summary_title) {
      await expect(page.locator('#search-q')).toHaveValue(
        new RegExp(ep.summary_title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
      )
    }
    // #671 — "Active advanced filters" summary region replaced by chip-active state. The "More" chip
    // increments its count when any advanced field is non-default; opening it shows the slim dialog.
    const moreChip = page.getByTestId('search-chip-more')
    await expect(moreChip).toContainText('More: 1')
    await moreChip.click()
    const advancedDialog = page.getByRole('dialog', { name: 'Advanced search' })
    await expect(advancedDialog).toBeVisible()
    await expect(advancedDialog.locator('#search-advanced-feed')).toHaveValue(feed.display_title)
  })

  /**
   * #1619 category B — still mocked; needs a v4 fixture, not a rewrite.
   *
   * This asserts the *empty* branch of the Similar panel. Against a real indexed corpus every
   * episode has nearest neighbours, so `items: []` is unreachable — the state has to be built,
   * not discovered. Recorded in docs/wip/CORPUS-V4-FIXTURE-LADDER.md §B.
   */
  test('similar empty state when API returns no peers', async ({ page }) => {
    await page.route('**/api/index/stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          available: true,
          reason: null,
          stats: {
            total_vectors: 1,
            doc_type_counts: {},
            feeds_indexed: ['f1'],
            embedding_model: 'mock',
            embedding_dim: 8,
            last_updated: '2024-01-01T00:00:00Z',
            index_size_bytes: 0,
          },
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
          query_used: 'Summary head Point one Point two',
          items: [],
          error: null,
          detail: null,
        }),
      })
    })
    await openLibrary(page)
    const ep = await liveFirstEpisode(page)
    await page.getByRole('button', { name: episodeRowName(ep) }).click()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: 'Key points' }),
    ).toBeVisible()
    await page.getByRole('button', { name: 'Episode and feed diagnostics' }).click()
    await expect(page.getByRole('tooltip')).toContainText('Feed in vector index')
    await expect(page.getByTestId('library-similar')).toBeVisible()
    await expect(page.getByTestId('library-similar-empty')).toBeVisible()
  })

  test('topic cluster checkbox adds topic_cluster_only=true to episodes request', async ({
    page,
  }) => {
    await openLibrary(page)
    const clusterReq = page.waitForRequest(
      (r) =>
        r.url().includes('/api/corpus/episodes') &&
        r.url().includes('topic_cluster_only=true'),
    )
    await page.getByTestId('library-topic-cluster-toggle').click()
    const req = await clusterReq
    expect(req.url()).toContain('topic_cluster_only=true')
  })

  test('row show name scopes Library to that feed (feed_id in episodes request)', async ({
    page,
  }) => {
    await openLibrary(page)
    const ep = await liveFirstEpisode(page)
    const scopedReq = page.waitForRequest(
      (r) =>
        r.url().includes('/api/corpus/episodes') &&
        r.url().includes(`feed_id=${ep.feed_id}`),
    )
    await page.getByTestId('library-row-scope-show').first().click()
    const req = await scopedReq
    expect(req.url()).toContain(`feed_id=${ep.feed_id}`)
  })

  /**
   * #1619 — blocked on the search index, not on assertions.
   *
   * The "why this episode" snippet is driven by an *active search context*, so the test has to
   * run a query and get results back. Migrating it needs a live `/api/search`; see the search
   * blocker recorded in docs/wip/CORPUS-V4-FIXTURE-LADDER.md §B.
   */
  test('active search context renders "why this episode" snippet on Library rows', async ({
    page,
  }) => {
    await openLibrary(page)
    const ep = await liveFirstEpisode(page)

    /* The rows are live, so the stubbed hit must name a LIVE episode_id — the snippet is matched
     * to a row by id. (Migration bite: a hand-written `e1` matches nothing once the Library is
     * real, and the row simply never renders a snippet.) */
    await page.route('**/api/search?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: 'climate',
          results: [
            {
              doc_id: `insight:${ep.episode_id}:1`,
              score: 0.93,
              text: 'Climate policy is the through-line of this episode.',
              metadata: { doc_type: 'insight', episode_id: ep.episode_id },
            },
          ],
        }),
      })
    })

    // No snippet before a search context is active.
    await expect(page.getByTestId('library-row-why')).toHaveCount(0)

    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await page.locator('#search-q').fill('climate')
    // Submit via Enter (SearchPanel handles Enter as submit); scoped locators
    // for the form-linked Search button are brittle — see person-landing.spec.
    await page.locator('#search-q').press('Enter')

    // §S4-shell pivot: search lives on the Search main tab; ``library-row-why``
    // lives on the Library tab. Switch back to Library to assert the snippet
    // (activeSearchContext is store-persisted across tab switches).
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()

    const why = page.getByTestId('library-row-why').first()
    await expect(why).toBeVisible()
    await expect(why).toContainText('Climate policy is the through-line')
  })

  test('FR4.3: Episode rail shows related insights from the relational layer', async ({
    page,
  }) => {
    await openLibrary(page)
    const ep = await liveFirstEpisode(page)
    /* The relational layer answers for real: assert the rail renders the insights the server
     * returns for THIS episode, however many that is. */
    const resp = await page.request.get(
      `/api/relational/episode-insights?episode=${encodeURIComponent(ep.episode_id)}`,
    )
    const { results } = (await resp.json()) as { results: { text: string }[] }
    expect(results.length).toBeGreaterThan(0)

    await page.getByRole('button', { name: episodeRowName(ep) }).click()
    const related = page.getByTestId('episode-related-insights')
    await expect(related).toBeVisible()
    /* Standard budget: the widened one was a mitigation for an APP bug, now fixed at cause.
     *
     * `loadRelatedInsights` bailed to an empty list when `corpusPath`/`healthStatus` were not ready,
     * and the watcher keyed on `episode_id` alone — so the bail was permanent and the request was
     * never made. This assertion is the regression guard: it failed with the server returning 20
     * results in ~50ms. See EpisodeDetailPanel.vue. */
    await expect(related.getByTestId('episode-related-insights-row').first()).toBeVisible()
    await expect(related).toContainText(results[0]!.text.slice(0, 40))
  })

  /**
   * #1619 category C — permanently mocked, correctly.
   *
   * This drives the `no_index` branch: a corpus whose vector index is missing. A live backend
   * serving an indexed corpus cannot produce it on demand, and no fixture version fixes that.
   */
  test('similar panel shows no-index message when API returns no_index', async ({ page }) => {
    await page.route('**/api/index/stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          available: true,
          reason: null,
          stats: {
            total_vectors: 0,
            doc_type_counts: {},
            feeds_indexed: [],
            embedding_model: 'mock',
            embedding_dim: 8,
            last_updated: '2024-01-01T00:00:00Z',
            index_size_bytes: 0,
          },
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
          query_used: 'Summary head Point one Point two',
          items: [],
          error: 'no_index',
          detail: null,
        }),
      })
    })
    await openLibrary(page)
    const ep = await liveFirstEpisode(page)
    await page.getByRole('button', { name: episodeRowName(ep) }).click()
    await expect(
      page.getByText('No vector index for this corpus yet', { exact: false }),
    ).toBeVisible({ timeout: 20_000 })
  })
})

/**
 * #1619 category B — a corpus SHAPE the v3 fixture cannot have, so this describe keeps the
 * catch-all stub (no `liveApi`).
 *
 * The feed-filter search input only renders above a 15-feed threshold and the v3 corpus ships
 * **9** feeds. It also has to run fully stubbed rather than half-live: 16 synthetic feeds against
 * a live episode list describe a corpus that does not exist, and the contradiction made the test
 * flaky (feed chip never settling) rather than failing honestly. Recorded in
 * docs/wip/CORPUS-V4-FIXTURE-LADDER.md §B.
 */
test.describe('Corpus Library tab — feed-count threshold (stubbed corpus)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test('feed list shows filter search when more than 15 feeds and filters client-side', async ({
    page,
  }) => {
    const feeds = Array.from({ length: 16 }, (_, i) => ({
      feed_id: `f${i + 1}`,
      display_title: `Library Mock Feed ${i + 1}`,
      episode_count: i === 0 ? 1 : 0,
    }))
    await page.route('**/api/health**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok', corpus_library_api: true, corpus_digest_api: true }),
      })
    })
    await page.route('**/api/corpus/feeds**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', feeds }),
      })
    })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await openLibraryFeedChip(page)
    const feedSearch = page.getByTestId('library-feed-filter-search')
    await expect(feedSearch).toBeVisible()
    await expect(
      page.getByRole('button', { name: 'Library Mock Feed 1, feed id f1, 1 episodes' }),
    ).toBeVisible()
    await feedSearch.fill('Library Mock Feed 16')
    await expect(
      page.getByRole('button', { name: 'Library Mock Feed 16, feed id f16, 0 episodes' }),
    ).toBeVisible()
    await expect(
      page.getByRole('button', { name: 'Library Mock Feed 1, feed id f1, 1 episodes' }),
    ).toHaveCount(0)
  })
})
