import { expect, test, type Page } from '@playwright/test'
import { setupDashboardApiMocks } from './dashboardApiMocks'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * UXS-006 §3.3–3.5 (briefing) and §4.1–4.3 (Coverage tab).
 *
 * Mocked on purpose: the briefing's action items are thresholds (GI < 50%, index missing/stale,
 * failed episodes) and the v3 fixture corpus sits on the all-clear side of every one of them
 * (40/40 GI, a fresh index, no run.json summaries). Each case below is a corpus STATE the live
 * fixture cannot be put into without rewriting it. The last describe runs on the live corpus,
 * because the defect it pins only shows against real artifact mtimes.
 */

const HOUR = 3_600_000
const DAY = 24 * HOUR

type RunOverrides = {
  created_at: string
  run_duration_seconds: number
  episodes_scraped_total: number
  episode_outcomes: Record<string, number>
}

function runSummary(o: RunOverrides) {
  return {
    relative_path: 'feeds/f-low/run_1/run.json',
    run_id: 'run_1',
    errors_total: 0,
    gi_artifacts_generated: null,
    kg_artifacts_generated: null,
    time_scraping_seconds: null,
    time_parsing_seconds: null,
    time_normalizing_seconds: null,
    time_io_and_waiting_seconds: null,
    ads_filtered_count: null,
    dialogue_insights_dropped_count: null,
    topics_normalized_count: null,
    entity_kinds_repaired_count: null,
    ad_chars_excised_preroll: null,
    ad_chars_excised_postroll: null,
    ad_episodes_with_excision_count: null,
    ...o,
  }
}

async function fulfillJson(page: Page, pattern: string, body: unknown): Promise<void> {
  await page.route(pattern, (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) }),
  )
}

const FEEDS = [
  { feed_id: 'f-low', display_title: 'Feed Low', episode_count: 5 },
  { feed_id: 'f-mid', display_title: 'Feed Mid', episode_count: 5 },
  { feed_id: 'f-high', display_title: 'Feed High', episode_count: 5 },
]

/** Corpus state shared by every test; each test then layers its own runs / coverage / index. */
async function setupCorpus(page: Page): Promise<void> {
  await mockSignIn(page, 'admin')
  await setupDashboardApiMocks(page)
  await fulfillJson(page, '**/api/corpus/feeds?**', { path: '/mock/corpus', feeds: FEEDS })
}

async function mockIndex(page: Page, available: boolean): Promise<void> {
  await fulfillJson(page, '**/api/index/stats**', {
    available,
    reason: available ? null : 'no_index',
    stats: available
      ? {
          total_vectors: 1200,
          doc_type_counts: { insight: 1200 },
          feeds_indexed: FEEDS.map((f) => f.feed_id),
          embedding_model: 'm',
          embedding_dim: 384,
          last_updated: new Date(Date.now() - HOUR).toISOString(),
          index_size_bytes: 1,
        }
      : null,
    reindex_recommended: false,
    reindex_reasons: [],
  })
}

async function mockCoverage(
  page: Page,
  c: { total: number; withGi: number; byFeed?: unknown[]; byMonth?: unknown[] },
): Promise<void> {
  await fulfillJson(page, '**/api/corpus/coverage?**', {
    path: '/mock/corpus',
    total_episodes: c.total,
    with_gi: c.withGi,
    with_kg: c.withGi,
    with_both: c.withGi,
    with_neither: c.total - c.withGi,
    by_month: c.byMonth ?? [],
    by_feed: c.byFeed ?? [],
  })
}

async function openDashboard(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill('/mock/corpus')
  await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
  await expect(page.getByTestId('briefing-card')).toBeVisible()
}

test.describe('Dashboard briefing (UXS-006 §3.3–3.5)', () => {
  test('last run shows status, duration and age; Details → opens Pipeline job history', async ({
    page,
  }) => {
    await setupCorpus(page)
    await mockIndex(page, true)
    await mockCoverage(page, { total: 10, withGi: 10 })
    await fulfillJson(page, '**/api/corpus/runs/summary?**', {
      path: '/mock/corpus',
      runs: [
        runSummary({
          created_at: new Date(Date.now() - 2 * HOUR - 5 * 60_000).toISOString(),
          run_duration_seconds: 125,
          episodes_scraped_total: 7,
          episode_outcomes: { ok: 7 },
        }),
      ],
    })
    await openDashboard(page)

    const lastRun = page.getByTestId('briefing-last-run')
    await expect(lastRun).toContainText('Success')
    await expect(lastRun).toContainText('7 episodes')
    await expect(lastRun).toContainText('2m 5s')
    await expect(lastRun).toContainText('2 hours ago')

    const tablist = page.getByRole('tablist', { name: 'Dashboard tabs' })
    await expect(tablist.getByRole('tab', { name: 'Coverage' })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    await page.getByTestId('briefing-last-run-details').click()
    await expect(tablist.getByRole('tab', { name: 'Pipeline' })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    await expect(page.getByTestId('dashboard-pipeline-subtab-job-history')).toHaveAttribute(
      'aria-selected',
      'true',
    )
  })

  test('healthy corpus: metrics render and the action list is the all-clear line', async ({
    page,
  }) => {
    await setupCorpus(page)
    await mockIndex(page, true)
    await mockCoverage(page, { total: 10, withGi: 10 })
    await fulfillJson(page, '**/api/corpus/runs/summary?**', {
      path: '/mock/corpus',
      runs: [
        runSummary({
          created_at: new Date(Date.now() - HOUR).toISOString(),
          run_duration_seconds: 30,
          episodes_scraped_total: 3,
          episode_outcomes: { ok: 3 },
        }),
      ],
    })
    await openDashboard(page)

    const health = page.getByTestId('briefing-corpus-health')
    await expect(health).toContainText('10 episodes')
    await expect(health).toContainText('100% with GI')
    await expect(health).toContainText('1.2k vectors')
    await expect(health).toContainText('3 feeds')

    await expect(page.getByTestId('briefing-all-clear')).toHaveText('● Everything looks good')
    await expect(page.getByTestId('briefing-action-item')).toHaveCount(0)
  })

  test('failures, low GI coverage and a missing index each produce an action item, worst first', async ({
    page,
  }) => {
    await setupCorpus(page)
    await mockIndex(page, false)
    await mockCoverage(page, { total: 10, withGi: 3 })
    await fulfillJson(page, '**/api/corpus/runs/summary?**', {
      path: '/mock/corpus',
      runs: [
        runSummary({
          created_at: new Date(Date.now() - HOUR).toISOString(),
          run_duration_seconds: 3700,
          episodes_scraped_total: 5,
          episode_outcomes: { ok: 3, failed: 2 },
        }),
      ],
    })
    await openDashboard(page)

    await expect(page.getByTestId('briefing-last-run')).toContainText('Partial')
    await expect(page.getByTestId('briefing-last-run')).toContainText('1h 1m 40s')
    await expect(page.getByTestId('briefing-corpus-health')).toContainText('30% with GI')

    const items = page.getByTestId('briefing-action-item')
    await expect(items).toHaveCount(3)
    await expect(items.nth(0)).toContainText('2 episodes failed in last run')
    await expect(items.nth(0)).toContainText('View failures')
    await expect(items.nth(1)).toContainText('7 episodes have no GI artifacts')
    await expect(items.nth(1)).toContainText('View in Library')
    await expect(items.nth(2)).toContainText('Vector index has not been built')
    await expect(items.nth(2)).toContainText('Open index controls')
    await expect(page.getByTestId('briefing-all-clear')).toHaveCount(0)

    // "View in Library" for the GI gap lands on Library asking only for episodes WITHOUT GI.
    const episodesReq = page.waitForRequest(
      (r) => r.url().includes('/api/corpus/episodes') && r.url().includes('has_gi=false'),
    )
    await items.nth(1).getByRole('button', { name: 'View in Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await episodesReq
  })

  test('episode count metric navigates to the Library', async ({ page }) => {
    await setupCorpus(page)
    await mockIndex(page, true)
    await mockCoverage(page, { total: 10, withGi: 10 })
    await openDashboard(page)

    await page
      .getByTestId('briefing-corpus-health')
      .getByRole('button', { name: '10 episodes' })
      .click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await expect(page.getByTestId('briefing-card')).toBeHidden()
  })
})

test.describe('Dashboard Coverage tab (UXS-006 §4.1–4.3)', () => {
  // Server order (`routes/corpus_coverage.py` sorts by GI ratio ascending) — the table must keep it.
  const BY_FEED = [
    { feed_id: 'f-low', display_title: 'Feed Low', total: 5, with_gi: 1, with_kg: 1 },
    { feed_id: 'f-mid', display_title: 'Feed Mid', total: 5, with_gi: 3, with_kg: 2 },
    { feed_id: 'f-high', display_title: 'Feed High', total: 5, with_gi: 5, with_kg: 5 },
  ]
  const BY_MONTH = [
    { month: '2024-01', total: 4, with_gi: 4, with_kg: 4, with_both: 4 },
    { month: '2024-02', total: 4, with_gi: 1, with_kg: 1, with_both: 1 },
    { month: '2024-03', total: 4, with_gi: 3, with_kg: 3, with_both: 3 },
  ]

  test.beforeEach(async ({ page }) => {
    await setupCorpus(page)
    await mockIndex(page, true)
    await mockCoverage(page, { total: 15, withGi: 9, byFeed: BY_FEED, byMonth: BY_MONTH })
    const day = (offset: number) => new Date(Date.now() - offset * DAY).toISOString()
    await fulfillJson(page, '**/api/artifacts?**', {
      path: '/mock/corpus',
      artifacts: [
        { name: 'a.gi.json', relative_path: 'a.gi.json', kind: 'gi', size_bytes: 10, mtime_utc: day(1), publish_date: '2024-03-01' },
        { name: 'a.kg.json', relative_path: 'a.kg.json', kind: 'kg', size_bytes: 10, mtime_utc: day(3), publish_date: '2024-03-01' },
      ],
    })
  })

  test('month chart and feed table (lowest GI first) render with their insight lines', async ({
    page,
  }) => {
    await openDashboard(page)

    const month = page.getByTestId('coverage-by-month-chart')
    await expect(month).toBeVisible()
    await expect(month.locator('canvas')).toBeVisible()
    await expect(month).toContainText('1 month below average — 2024-02 need attention')

    const table = page.getByTestId('feed-coverage-table')
    const rows = table.getByTestId('feed-coverage-row')
    await expect(rows).toHaveCount(3)
    await expect(rows.nth(0)).toContainText('Feed Low')
    await expect(rows.nth(0)).toContainText('20%')
    await expect(rows.nth(1)).toContainText('60%')
    await expect(rows.nth(2)).toContainText('100%')
    await expect(table).toContainText(
      'Feed Feed Low has lowest GI coverage at 20% — 4 episodes without GI artifacts.',
    )
  })

  /**
   * The chart buckets `shell.artifactList`, which the Dashboard never fetches itself (see the
   * live test below). Listing artifacts from the status bar first is a real user path that fills
   * it, so these two exercise the bucketing + insight copy rather than an always-empty list.
   */
  async function listArtifactsThenOpenDashboard(page: Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await page.getByTestId('status-bar-list-artifacts').click()
    await expect(page.getByTestId('artifact-list-dialog')).toBeVisible()
    await page.getByTestId('artifact-list-close').click()
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await expect(page.getByTestId('briefing-card')).toBeVisible()
  }

  test('artifact activity names the newest GI and KG days', async ({ page }) => {
    await listArtifactsThenOpenDashboard(page)
    const activity = page.getByTestId('artifact-activity-chart')
    await expect(activity.locator('canvas')).toBeVisible()
    const ymd = (offset: number) => new Date(Date.now() - offset * DAY).toISOString().slice(0, 10)
    await expect(activity).toContainText(`Last GI: ${ymd(1)} · Last KG: ${ymd(3)}`)
  })

  test('artifact activity calls out 14 silent days', async ({ page }) => {
    await fulfillJson(page, '**/api/artifacts?**', {
      path: '/mock/corpus',
      artifacts: [
        {
          name: 'old.gi.json',
          relative_path: 'old.gi.json',
          kind: 'gi',
          size_bytes: 10,
          mtime_utc: new Date(Date.now() - 20 * DAY).toISOString(),
          publish_date: '2024-03-01',
        },
      ],
    })
    await listArtifactsThenOpenDashboard(page)
    await expect(page.getByTestId('artifact-activity-chart')).toContainText(
      'No new artifacts in 14 days — pipeline may not be running',
    )
  })

  test('clicking a feed row opens the Library scoped to that feed', async ({ page }) => {
    await openDashboard(page)
    const episodesReq = page.waitForRequest(
      (r) => r.url().includes('/api/corpus/episodes') && r.url().includes('feed_id=f-mid'),
    )
    await page.getByTestId('feed-coverage-row').nth(1).getByText('Feed Mid').click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await expect(page.getByTestId('library-chip-feed')).toContainText('Feed: Feed Mid')
    await episodesReq
  })
})

test.describe('Dashboard Coverage tab — live corpus', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin', { liveApi: true })
  })

  /**
   * REAL BUG — kept asserting the correct behaviour, marked `test.fail()`.
   *
   * UXS-006 §4.3: the chart's source is `GET /api/artifacts` mtimes. But `ArtifactActivityChart`
   * renders `shell.artifactList`, and nothing on the Dashboard path fetches it — only the Graph
   * tab's corpus sync, the status-bar **List** button and the search "On graph" handoff call
   * `shell.fetchArtifactList()`. Open the Dashboard first (the operator's normal landing) on a
   * corpus whose artifacts were all written today and the card reads
   * "No new artifacts in 14 days — pipeline may not be running": a false alarm, on the panel
   * whose stated job is making silence visible. Remove `test.fail()` once the Dashboard loads
   * the artifact list itself.
   */
  test('artifact activity reflects the corpus artifacts when the Dashboard is opened first', async ({
    page,
  }) => {
    test.fail()
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    const root = await liveCorpusRoot(page)
    const resp = await page.request.get(`/api/artifacts?path=${encodeURIComponent(root)}`)
    const { artifacts } = (await resp.json()) as {
      artifacts: { kind: string; mtime_utc: string }[]
    }
    const newest = (kind: string) =>
      artifacts
        .filter((a) => a.kind === kind)
        .map((a) => a.mtime_utc.slice(0, 10))
        .sort()
        .at(-1)
    await statusBarCorpusPathInput(page).fill(root)
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await expect(page.getByTestId('briefing-card')).toBeVisible()
    await expect(page.getByTestId('artifact-activity-chart')).toContainText(
      `Last GI: ${newest('gi')} · Last KG: ${newest('kg')}`,
      { timeout: 10_000 },
    )
  })
})
