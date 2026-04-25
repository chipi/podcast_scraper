import { expect, test, type Page } from '@playwright/test'
import { setupDashboardApiMocks, setupCorpusDashboardDataRoutes } from './dashboardApiMocks'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

/**
 * #656 Stage B + Stage D — behaviour tests for the two pipeline metrics
 * panels surfaced on the dashboard:
 *   - PipelineCleanupMetrics: 4 post-extraction filter counters (#652)
 *   - PipelineAdExcisionMetrics: 3 pre-extraction ad-excision counters
 *     (#663 + #656 Stage D)
 *
 * Both panels read from ``CorpusRunSummaryItem`` via
 * ``/api/corpus/runs/summary``. Each counter renders as:
 *   - real number > 0 → localised value with an info badge
 *   - 0 → "0" (no badge)
 *   - null → "—" (legacy run, no badge)
 *
 * The vitest source-guards cover the component-local invariants.
 * These Playwright tests cover the end-to-end wiring:
 *   request → schema → TS interface → component → DOM.
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

async function mockDashboardWithRuns(
  page: Page,
  runs: Array<Record<string, unknown>>,
): Promise<void> {
  await setupDashboardApiMocks(page)
  await setupCorpusDashboardDataRoutes(page)
  // Override the default empty-runs mock with our fixture.
  await mockRunSummary(page, runs)
  await page.route('**/api/index/stats**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        available: true,
        stats: { total_vectors: 0, doc_type_counts: {} },
        reindex_recommended: false,
        reindex_reasons: [],
      }),
    })
  })
}

const POPULATED_RUN = {
  relative_path: 'run_20260424.json',
  run_id: 'run_20260424',
  created_at: '2026-04-24T00:00:00Z',
  episode_outcomes: { ok: 5 },
  // #656 Stage B counters
  ads_filtered_count: 12,
  dialogue_insights_dropped_count: 3,
  topics_normalized_count: 0,
  entity_kinds_repaired_count: 8,
  // #656 Stage D counters
  ad_chars_excised_preroll: 1250,
  ad_chars_excised_postroll: 0,
  ad_episodes_with_excision_count: 4,
}

const LEGACY_RUN = {
  relative_path: 'run_legacy.json',
  run_id: 'run_legacy',
  created_at: '2024-01-01T00:00:00Z',
  episode_outcomes: { ok: 1 },
  // All #652/#663 counters omitted → null in the schema → "—" in the UI
}

async function navigateToPipelineTab(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill('/mock/corpus')
  await statusBarCorpusPathInput(page).press('Enter')
  await mainViewsNav(page).getByRole('button', { name: /^dashboard$/i }).click()
}

test.describe('Dashboard pipeline metrics (mocked API)', () => {
  test.describe.configure({ mode: 'serial' })

  test('populated run renders Stage B cleanup counters + Stage D ad-excision counters', async ({ page }) => {
    await mockDashboardWithRuns(page, [POPULATED_RUN])
    await navigateToPipelineTab(page)

    // The two panels live on the Intelligence sub-tab of Dashboard (mounted
    // alongside TopVoices / TopicLandscape). The default sub-tab is
    // Coverage; click Intelligence to reveal them.
    await page.getByRole('tab', { name: /^intelligence$/i }).click()

    const cleanup = page.getByTestId('pipeline-cleanup-metrics')
    await expect(cleanup).toBeVisible()
    await expect(cleanup.getByTestId('pipeline-cleanup-ads')).toContainText('12')
    await expect(cleanup.getByTestId('pipeline-cleanup-dialogue')).toContainText('3')
    await expect(cleanup.getByTestId('pipeline-cleanup-topics')).toContainText('0')
    await expect(cleanup.getByTestId('pipeline-cleanup-entities')).toContainText('8')

    const excision = page.getByTestId('pipeline-ad-excision-metrics')
    await expect(excision).toBeVisible()
    await expect(excision.getByTestId('pipeline-ad-excision-preroll')).toContainText('1,250')
    await expect(excision.getByTestId('pipeline-ad-excision-postroll')).toContainText('0')
    await expect(excision.getByTestId('pipeline-ad-excision-episodes')).toContainText('4')
  })

  test('legacy run (all counters null) renders "—" in every row, no info chips', async ({ page }) => {
    await mockDashboardWithRuns(page, [LEGACY_RUN])
    await navigateToPipelineTab(page)
    await page.getByRole('tab', { name: /^intelligence$/i }).click()

    const cleanup = page.getByTestId('pipeline-cleanup-metrics')
    await expect(cleanup).toBeVisible()
    // All four cleanup rows show the em-dash.
    for (const id of [
      'pipeline-cleanup-ads',
      'pipeline-cleanup-dialogue',
      'pipeline-cleanup-topics',
      'pipeline-cleanup-entities',
    ]) {
      await expect(cleanup.getByTestId(id)).toContainText('—')
    }

    const excision = page.getByTestId('pipeline-ad-excision-metrics')
    await expect(excision).toBeVisible()
    for (const id of [
      'pipeline-ad-excision-preroll',
      'pipeline-ad-excision-postroll',
      'pipeline-ad-excision-episodes',
    ]) {
      await expect(excision.getByTestId(id)).toContainText('—')
    }

    // The "predates #652 / #663" help copy appears when every counter
    // is null.
    await expect(cleanup).toContainText(/predates the #652/i)
    await expect(excision).toContainText(/predates the #663/i)
  })
})
