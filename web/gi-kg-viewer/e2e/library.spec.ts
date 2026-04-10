import { expect, test, type Page } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE } from './helpers'

async function expandLibraryEpisodeFilters(page: Page): Promise<void> {
  const btn = page.getByRole('button', { name: /^Episode filters/i })
  if ((await btn.getAttribute('aria-expanded')) === 'false') {
    await btn.click()
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

  test('lists mocked feeds and episodes; search handoff fills query and feed filter', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await expandLibraryEpisodeFilters(page)
    await expect(
      page.getByRole('button', { name: 'Mock Show, feed id f1, 1 episodes' }),
    ).toBeVisible()
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
    await page.getByRole('button', { name: 'Prefill semantic search' }).click()
    await expect(page.locator('#search-q')).toHaveValue('Short prose summary for tests.')
    await expect(
      page.getByText(/From Library: query uses episode summary/i),
    ).toBeVisible()
    await expect(
      page.locator('label').filter({ hasText: 'Feed id (substring)' }).locator('input'),
    ).toHaveValue('f1')
  })

  test('shows indexed badge, similar empty state when API returns no peers', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await expandLibraryEpisodeFilters(page)
    await expect(
      page.getByTestId('library-root').getByText('Indexed', { exact: true }),
    ).toBeVisible()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: 'Key points' }),
    ).toBeVisible()
    await page.getByRole('button', { name: 'Episode and feed diagnostics' }).click()
    await expect(page.getByRole('tooltip')).toContainText('Feed in vector index')
    await expect(page.getByTestId('library-similar')).toBeVisible()
    await page.getByRole('button', { name: 'Find similar episodes' }).click()
    await expect(page.getByTestId('library-similar-empty')).toBeVisible()
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
    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await expect(
      page.getByTestId('library-root').getByText('Indexed', { exact: true }),
    ).toHaveCount(0)
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Find similar episodes' }).click()
    await expect(
      page.getByText('No vector index for this corpus yet', { exact: false }),
    ).toBeVisible()
  })
})
