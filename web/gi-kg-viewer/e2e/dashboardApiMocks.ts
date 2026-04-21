import type { Page } from '@playwright/test'

/** Corpus + digest routes used by Dashboard (no health / list-artifacts). Register after your own GET /api/artifacts mock so listing stays corpus-specific. */
export async function setupCorpusDashboardDataRoutes(page: Page): Promise<void> {
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
  await page.route('**/api/corpus/coverage?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        total_episodes: 12,
        with_gi: 10,
        with_kg: 8,
        with_both: 7,
        with_neither: 2,
        by_month: [{ month: '2024-03', total: 2, with_gi: 2, with_kg: 1, with_both: 1 }],
        by_feed: [
          {
            feed_id: 'f1',
            display_title: 'Feed One',
            total: 5,
            with_gi: 4,
            with_kg: 3,
          },
        ],
      }),
    })
  })
  await page.route('**/api/corpus/persons/top?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ path: '/mock/corpus', persons: [], total_persons: 0 }),
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
  await page.route('**/api/corpus/digest**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        window: '7d',
        window_start_utc: '2024-01-01T00:00:00Z',
        window_end_utc: '2024-01-02T00:00:00Z',
        compact: false,
        rows: [],
        topics: [],
        topics_unavailable_reason: null,
      }),
    })
  })
}

/** Minimal API stubs so Dashboard loads without a real corpus server. */
export async function setupDashboardApiMocks(page: Page): Promise<void> {
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
            mtime_utc: '2026-04-18T12:00:00Z',
            publish_date: '2026-04-18',
          },
        ],
      }),
    })
  })
  await setupCorpusDashboardDataRoutes(page)
}
