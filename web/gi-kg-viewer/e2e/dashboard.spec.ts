import { expect, test } from '@playwright/test'
import { leftPanelTabs, loadGraphViaFilePicker, mainViewsNav, SHELL_HEADING_RE } from './helpers'

test.describe('Dashboard tab', () => {
  test('shows charts row; data overview lives under API · Data', async ({ page }) => {
    await loadGraphViaFilePicker(page)

    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()

    await expect(
      page.getByText(
        /Corpus root, catalog snapshot, graph metrics, and vector index live under[\s\S]*API · Data[\s\S]*left panel/,
      ),
    ).toBeVisible()
  })

  test('shows Data section in left panel after graph is loaded', async ({ page }) => {
    await loadGraphViaFilePicker(page)

    await leftPanelTabs(page).getByRole('button', { name: 'API · Data' }).click()

    await expect(
      page.getByRole('heading', { name: 'Data', exact: true }),
    ).toBeVisible()
    await expect(
      page.getByText(/Corpus root, catalog snapshot from/i),
    ).toBeVisible()
    await expect(
      page.getByRole('heading', { name: 'Corpus root', level: 3 }),
    ).toBeVisible()
    await expect(
      page.getByRole('heading', { name: 'Corpus catalog', level: 3 }),
    ).toBeVisible()
    await expect(
      page.getByRole('heading', { name: 'Graph', level: 3 }),
    ).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Vector index' })).toBeVisible()
  })

  test('shows corpus summary counts when API and corpus path are available', async ({ page }) => {
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
          artifacts: [
            {
              name: 'a.gi.json',
              relative_path: 'a.gi.json',
              kind: 'gi',
              size_bytes: 10,
              mtime_utc: '2024-01-01T00:00:00Z',
            },
          ],
        }),
      })
    })
    await page.route('**/api/corpus/stats?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          publish_month_histogram: { '2024-03': 2 },
          catalog_episode_count: 12,
          catalog_feed_count: 3,
          digest_topics_configured: 2,
        }),
      })
    })
    await page.route('**/api/corpus/runs/summary?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', runs: [] }),
      })
    })
    await page.route('**/api/corpus/feeds?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', feeds: [] }),
      })
    })
    await page.route('**/api/index/stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          available: true,
          index_path: '/mock/corpus/.idx',
          stats: {
            total_vectors: 100,
            doc_type_counts: { episode: 100 },
            feeds_indexed: ['f1', 'f2'],
            embedding_model: 'mock-model',
            embedding_dim: 4,
            last_updated: '2024-01-01T12:00:00Z',
            index_size_bytes: 4096,
          },
          reindex_recommended: false,
          reindex_reasons: [],
          artifact_newest_mtime: '2024-01-01T14:00:00Z',
          search_root_hints: [],
          rebuild_in_progress: false,
          rebuild_last_error: null,
        }),
      })
    })
    await page.route('**/api/corpus/digest**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          window: '24h',
          window_start_utc: '2024-01-01T00:00:00Z',
          window_end_utc: '2024-01-02T00:00:00Z',
          compact: true,
          rows: [],
          topics: [],
          topics_unavailable_reason: null,
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()

    const strip = page.getByRole('group', { name: 'Corpus summary counts' })
    await expect(strip).toBeVisible({ timeout: 15_000 })
    const nums = strip.locator('.tabular-nums')
    await expect(nums).toHaveCount(4)
    await expect(nums.nth(0)).toHaveText('3')
    await expect(nums.nth(1)).toHaveText('12')
    await expect(nums.nth(2)).toHaveText('2')
    await expect(nums.nth(3)).toHaveText('1')
  })

  test('Dashboard sections: Pipeline default; Content intelligence shows catalog charts', async ({
    page,
  }) => {
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
          artifacts: [
            {
              name: 'a.gi.json',
              relative_path: 'a.gi.json',
              kind: 'gi',
              size_bytes: 10,
              mtime_utc: '2024-01-01T00:00:00Z',
            },
          ],
        }),
      })
    })
    await page.route('**/api/corpus/stats?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          publish_month_histogram: { '2024-03': 2 },
          catalog_episode_count: 12,
          catalog_feed_count: 3,
          digest_topics_configured: 2,
        }),
      })
    })
    await page.route('**/api/corpus/runs/summary?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', runs: [] }),
      })
    })
    await page.route('**/api/corpus/feeds?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', feeds: [] }),
      })
    })
    await page.route('**/api/index/stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          available: true,
          index_path: '/mock/corpus/.idx',
          stats: {
            total_vectors: 100,
            doc_type_counts: { episode: 100 },
            feeds_indexed: ['f1', 'f2'],
            embedding_model: 'mock-model',
            embedding_dim: 4,
            last_updated: '2024-01-01T12:00:00Z',
            index_size_bytes: 4096,
          },
          reindex_recommended: false,
          reindex_reasons: [],
          artifact_newest_mtime: '2024-01-01T14:00:00Z',
          search_root_hints: [],
          rebuild_in_progress: false,
          rebuild_last_error: null,
        }),
      })
    })
    await page.route('**/api/corpus/digest**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          window: '24h',
          window_start_utc: '2024-01-01T00:00:00Z',
          window_end_utc: '2024-01-02T00:00:00Z',
          compact: true,
          rows: [],
          topics: [],
          topics_unavailable_reason: null,
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()

    const tablist = page.getByRole('tablist', { name: 'Dashboard sections' })
    await expect(tablist).toBeVisible({ timeout: 15_000 })
    await expect(tablist.getByRole('tab', { name: 'Pipeline' })).toHaveAttribute('aria-selected', 'true')
    await expect(tablist.getByRole('tab', { name: 'Content intelligence' })).toHaveAttribute(
      'aria-selected',
      'false',
    )

    await tablist.getByRole('tab', { name: 'Content intelligence' }).click()
    await expect(tablist.getByRole('tab', { name: 'Content intelligence' })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    await expect(
      page.getByRole('heading', { name: 'Episodes by publish month (catalog)' }),
    ).toBeVisible()

    await expect(
      page.getByRole('region', { name: 'Vector index and digest glance' }),
    ).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Vector index and digest glance' })).toBeVisible()
    await expect(page.getByText(/Digest \(24h, compact\):/)).toBeVisible()
    await expect(
      page.getByText(/Catalog reports 12 episodes total; bars sum to 2/),
    ).toBeVisible()
  })
})
