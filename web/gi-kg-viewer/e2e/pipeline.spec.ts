import { expect, test, type Page } from '@playwright/test'
import { setupDashboardApiMocks, setupCorpusDashboardDataRoutes } from './dashboardApiMocks'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * UXS-006 §6 — the four canonical Pipeline-tab panels ship with these testids:
 *
 *   - ``pipeline-duration-trend``       (5-run VerticalBarChart wrapper)
 *   - ``pipeline-episodes-per-run``     (up-to-48-run VerticalBarChart wrapper)
 *   - ``pipeline-run-dot``              (per-row outcome bar in the Run
 *                                        history strip's listbox)
 *   - ``pipeline-feed-history-grid``    (multi-feed heatmap; hides on
 *                                        single-feed or legacy corpora)
 *
 * All four read from ``GET /api/corpus/runs/summary``. This spec drives
 * the surface end-to-end against mocked runs data.
 */

async function mockRunSummary(
  page: Page,
  runs: Array<Record<string, unknown>>,
): Promise<void> {
  await page.route('**/api/corpus/runs/summary?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ path: '/mock/corpus', runs }),
    })
  })
}

async function mockCorpusFeeds(
  page: Page,
  feeds: Array<Record<string, unknown>>,
): Promise<void> {
  await page.route('**/api/corpus/feeds?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ path: '/mock/corpus', feeds }),
    })
  })
}

function makeRun(
  feedId: string | null,
  createdAt: string,
  ok: number,
  failed = 0,
): Record<string, unknown> {
  return {
    relative_path: `feeds/${feedId ?? 'legacy'}/run_${createdAt}/run.json`,
    run_id: `${feedId ?? 'legacy'}-${createdAt}`,
    created_at: createdAt,
    run_duration_seconds: 120 + ok * 5,
    episodes_scraped_total: ok + failed,
    errors_total: failed,
    gi_artifacts_generated: ok,
    kg_artifacts_generated: ok,
    time_scraping_seconds: 30,
    time_parsing_seconds: 5,
    time_normalizing_seconds: 3,
    time_io_and_waiting_seconds: 10,
    episode_outcomes: { ok, failed, skipped: 0 },
    ads_filtered_count: null,
    dialogue_insights_dropped_count: null,
    topics_normalized_count: null,
    entity_kinds_repaired_count: null,
    ad_chars_excised_preroll: null,
    ad_chars_excised_postroll: null,
    ad_episodes_with_excision_count: null,
    feed_id: feedId,
  }
}

async function navigateToPipelineTab(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill('/mock/corpus')
  await statusBarCorpusPathInput(page).press('Enter')
  await mainViewsNav(page).getByRole('button', { name: /^dashboard$/i }).click()
  const tablist = page.getByRole('tablist', { name: 'Dashboard tabs' })
  await expect(tablist).toBeVisible({ timeout: 15_000 })
  await tablist.getByRole('tab', { name: /^pipeline$/i }).click()
}

const MULTI_FEED_RUNS = [
  makeRun('rss_alpha', '2026-07-01T12:00:00Z', 3, 0),
  makeRun('rss_beta', '2026-07-01T12:00:00Z', 2, 1),
  makeRun('rss_alpha', '2026-07-02T12:00:00Z', 4, 0),
  makeRun('rss_beta', '2026-07-02T12:00:00Z', 3, 0),
  makeRun('rss_alpha', '2026-07-03T12:00:00Z', 5, 0),
  makeRun('rss_beta', '2026-07-03T12:00:00Z', 4, 0),
]

const MULTI_FEED_CATALOG = [
  { feed_id: 'rss_alpha', display_title: 'Alpha Feed', episode_count: 12, rss_url: null, description: null },
  { feed_id: 'rss_beta', display_title: 'Beta Feed', episode_count: 9, rss_url: null, description: null },
]

test.describe('Dashboard — Pipeline tab canonical testids (UXS-006 §6)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin')
    await setupDashboardApiMocks(page)
    await setupCorpusDashboardDataRoutes(page)
  })

  test('duration trend, episodes-per-run, and run-dot render on the Pipeline tab', async ({
    page,
  }) => {
    await mockRunSummary(page, MULTI_FEED_RUNS)
    await mockCorpusFeeds(page, MULTI_FEED_CATALOG)
    await navigateToPipelineTab(page)

    // Duration trend: wraps the shared VerticalBarChart bound to durationFive.
    await expect(page.getByTestId('pipeline-duration-trend')).toBeVisible({
      timeout: 15_000,
    })

    // Episodes-per-run: wraps the shared VerticalBarChart bound to episodesPerRun.
    await expect(page.getByTestId('pipeline-episodes-per-run')).toBeVisible()

    // Run history sub-tab surfaces the listbox with per-row pipeline-run-dot bars.
    await page.getByTestId('dashboard-pipeline-subtab-history').click()
    const runDots = page.getByTestId('pipeline-run-dot')
    await expect(runDots.first()).toBeVisible()
    // Six mocked runs -> six outcome-bar dots in the strip.
    await expect(runDots).toHaveCount(6)
  })

  test('feed-history-grid renders on multi-feed corpora and hides on single-feed', async ({
    page,
  }) => {
    await mockRunSummary(page, MULTI_FEED_RUNS)
    await mockCorpusFeeds(page, MULTI_FEED_CATALOG)
    await navigateToPipelineTab(page)

    const grid = page.getByTestId('pipeline-feed-history-grid')
    await expect(grid).toBeVisible({ timeout: 15_000 })
    // 2 feeds x 3 distinct run-days = 6 cells.
    await expect(grid.getByTestId('pipeline-feed-history-cell')).toHaveCount(6)
    // Row headers hydrate from the /api/corpus/feeds catalog display_title.
    await expect(grid.getByRole('rowheader', { name: 'Alpha Feed' })).toBeVisible()
    await expect(grid.getByRole('rowheader', { name: 'Beta Feed' })).toBeVisible()
  })

  test('feed-history-grid silently hides when only one feed_id is present', async ({
    page,
  }) => {
    const singleFeed = MULTI_FEED_RUNS.filter((r) => r.feed_id === 'rss_alpha')
    await mockRunSummary(page, singleFeed)
    await mockCorpusFeeds(page, [MULTI_FEED_CATALOG[0]!])
    await navigateToPipelineTab(page)

    await expect(page.getByTestId('pipeline-duration-trend')).toBeVisible({
      timeout: 15_000,
    })
    await expect(page.getByTestId('pipeline-feed-history-grid')).toHaveCount(0)
  })

  test('feed-history-grid silently hides when runs carry no feed_id (legacy corpus)', async ({
    page,
  }) => {
    const legacy = MULTI_FEED_RUNS.map((r) => ({ ...r, feed_id: null }))
    await mockRunSummary(page, legacy)
    await mockCorpusFeeds(page, [])
    await navigateToPipelineTab(page)

    await expect(page.getByTestId('pipeline-duration-trend')).toBeVisible({
      timeout: 15_000,
    })
    await expect(page.getByTestId('pipeline-feed-history-grid')).toHaveCount(0)
  })
})
