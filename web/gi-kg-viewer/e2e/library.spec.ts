import { expect, test, type Page } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

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

test.describe('Corpus Library tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
        }),
      })
    })
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          artifacts: [],
        }),
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
              topics: ['Point one', 'Point two'],
              summary_title: 'Summary head',
              summary_bullets_preview: ['Point one', 'Point two'],
              summary_preview: 'Summary head — Point one · Point two',
              episode_id: 'e1',
              episode_title: 'Mock Episode Title',
              publish_date: '2024-06-01',
              gi_relative_path: 'metadata/ep1.gi.json',
              kg_relative_path: 'metadata/ep1.kg.json',
              has_gi: true,
              has_kg: true,
              cil_digest_topics: [],
            },
          ],
          next_cursor: null,
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
          episode_id: 'e1',
          episode_title: 'Mock Episode Title',
          publish_date: '2024-06-01',
          summary_title: 'Summary head',
          summary_bullets: ['Point one', 'Point two'],
          summary_text: 'Short prose summary for tests.',
          gi_relative_path: 'metadata/ep1.gi.json',
          kg_relative_path: 'metadata/ep1.kg.json',
          has_gi: true,
          has_kg: true,
          cil_digest_topics: [
            {
              topic_id: 'topic:alpha',
              label: 'Alpha topic',
              in_topic_cluster: true,
              topic_cluster_compound_id: 'tc:mock',
            },
            {
              topic_id: 'topic:beta',
              label: 'Beta topic',
              in_topic_cluster: false,
              topic_cluster_compound_id: null,
            },
          ],
        }),
      })
    })
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
  })

  test('Episode subject rail: slash focuses search; episode rail stays visible', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    // #669 — filter chips are always visible; no expand step needed.
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: 'Mock Episode Title' }),
    ).toBeVisible()
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('/')
    await expect(page.locator('#search-q')).toBeFocused()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: 'Mock Episode Title' }),
    ).toBeVisible()
  })

  test('feed filter: Clear feed filter resets to all feeds', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    // #669 — Clear lives inside the LibraryFeedChip popover and only renders
    // when a feed is selected (no disabled state in chip popover).
    await openLibraryFeedChip(page)
    const popover = page.getByTestId('library-popover-feed')
    await expect(popover.getByTestId('corpus-feed-filter-clear')).toHaveCount(0)
    await page.getByRole('button', { name: 'Mock Show, feed id f1, 1 episodes' }).click()
    await openLibraryFeedChip(page)
    const clearFeed = popover.getByTestId('corpus-feed-filter-clear')
    await expect(clearFeed).toBeVisible()
    await clearFeed.click()
    await openLibraryFeedChip(page)
    await expect(popover.getByTestId('corpus-feed-filter-clear')).toHaveCount(0)
  })

  test('feed list shows filter search when more than 15 feeds and filters client-side', async ({
    page,
  }) => {
    const feeds = Array.from({ length: 16 }, (_, i) => ({
      feed_id: `f${i + 1}`,
      display_title: `Library Mock Feed ${i + 1}`,
      episode_count: i === 0 ? 1 : 0,
    }))
    await page.unroute('**/api/corpus/feeds**')
    await page.route('**/api/corpus/feeds**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          feeds,
        }),
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

  test('lists mocked feeds and episodes; search handoff fills query and feed filter', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await openLibraryFeedChip(page)
    await expect(
      page.getByRole('button', { name: 'Mock Show, feed id f1, 1 episodes' }),
    ).toBeVisible()
    // Close the popover before clicking an episode behind it.
    await page.getByTestId('library-chip-feed').click()
    await expect(
      page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }),
    ).toBeVisible()
    await expect(
      page
        .getByTestId('library-root')
        .getByText('Summary head — Point one · Point two')
        .first(),
    ).toBeVisible()
    await expect(
      page.getByText('Summary head — Point one · Point two'),
    ).toBeVisible()
    await expect(page.getByText('Short prose summary for tests.')).toBeVisible()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: 'Key points' }),
    ).toBeVisible()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .locator('li')
        .filter({ hasText: /^Point one$/ }),
    ).toBeVisible()
    await expect(
      page
        .getByTestId('episode-detail-cil-pills')
        .getByRole('button', { name: 'Open graph for topic: Alpha topic' }),
    ).toBeVisible()
    await page.getByRole('button', { name: 'Prefill semantic search' }).click()
    // Same field order as Similar episodes / server build_similarity_query (title + bullets), not prose summary_text.
    await expect(page.locator('#search-q')).toHaveValue('Summary head Point one Point two')
    // #671 — "Active advanced filters" summary region replaced by chip-active state. The "More" chip
    // increments its count when any advanced field is non-default; opening it shows the slim dialog.
    const moreChip = page.getByTestId('search-chip-more')
    await expect(moreChip).toContainText('More: 1')
    await moreChip.click()
    const advancedDialog = page.getByRole('dialog', { name: 'Advanced search' })
    await expect(advancedDialog).toBeVisible()
    await expect(advancedDialog.locator('#search-advanced-feed')).toHaveValue('Mock Show')
  })

  test('similar empty state when API returns no peers', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    // #669 — filter chips are always visible; no expand step needed.
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
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
    const episodesPayload = {
      path: '/mock/corpus',
      feed_id: null,
      items: [
        {
          metadata_relative_path: 'metadata/ep1.metadata.json',
          feed_id: 'f1',
          feed_display_title: 'Mock Show',
          topics: ['Point one', 'Point two'],
          summary_title: 'Summary head',
          summary_bullets_preview: ['Point one', 'Point two'],
          summary_preview: 'Summary head — Point one · Point two',
          episode_id: 'e1',
          episode_title: 'Mock Episode Title',
          publish_date: '2024-06-01',
          gi_relative_path: 'metadata/ep1.gi.json',
          kg_relative_path: 'metadata/ep1.kg.json',
          has_gi: true,
          has_kg: true,
          cil_digest_topics: [],
        },
      ],
      next_cursor: null,
    }
    await page.unroute('**/api/corpus/episodes**')
    await page.route('**/api/corpus/episodes**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(episodesPayload),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    const clusterReq = page.waitForRequest(
      (r) =>
        r.url().includes('/api/corpus/episodes') &&
        r.url().includes('topic_cluster_only=true'),
    )
    await page.getByTestId('library-chip-clustered').click()
    const req = await clusterReq
    expect(req.url()).toContain('topic_cluster_only=true')
  })

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
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await expect(
      page.getByText('No vector index for this corpus yet', { exact: false }),
    ).toBeVisible({ timeout: 20_000 })
  })
})
