import { expect, test } from '@playwright/test'
import { setupDashboardApiMocks } from './dashboardApiMocks'
import {
  liveCorpusRoot,
  loadGraphViaFilePicker,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * #1619 — migrated to the live API, except where noted per test.
 *
 * The fixture corpus serves every Dashboard read this file needs: `/api/corpus/{stats,coverage,
 * persons/top,digest,topic-clusters,query-activity}` all return real data for the 36-episode v3
 * corpus. Two tests keep mocks, each for a reason recorded at the test.
 */
test.describe('Dashboard tab', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin', { liveApi: true })
  })

  test('briefing shows no-corpus empty state when path is unset', async ({ page }) => {
    await page.addInitScript(() => {
      try {
        localStorage.removeItem('ps_corpus_path')
      } catch {
        /* ignore */
      }
    })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await expect(page.getByTestId('briefing-no-corpus')).toBeVisible({ timeout: 15_000 })
    await expect(page.getByTestId('briefing-no-corpus')).toContainText(
      'Set a corpus path in the status bar below to begin.',
    )
    await expect(page.locator('[data-testid="briefing-last-run"]')).toHaveCount(0)
    await expect(page.locator('[data-testid="briefing-corpus-health"]')).toHaveCount(0)
    await expect(page.locator('[data-testid="briefing-action-items"]')).toHaveCount(0)
  })

  test('shows briefing card after opening Dashboard', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await expect(page.getByTestId('briefing-card')).toBeVisible({ timeout: 15_000 })
  })

  test('Coverage tab is default; Pipeline tab can be selected', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    // Wait for the Dashboard's DATA, not just its chrome. The tablist paints before the briefing
    // load settles, and the render that lands when it does replaces these buttons — so a click
    // fired in that window is discarded and the tab never becomes selected. Playwright's
    // actionability checks cannot see it: the button is visible, enabled and stable, it simply
    // gets thrown away. Measured on main 2026-08-18: 23 polls, every one `aria-selected="false"`,
    // then green on retry in 2.6 s. The sibling test above already waits for `briefing-card`;
    // this one did not, which is the whole difference.
    await expect(page.getByTestId('briefing-card')).toBeVisible({ timeout: 15_000 })

    const tablist = page.getByRole('tablist', { name: 'Dashboard tabs' })
    await expect(tablist).toBeVisible({ timeout: 15_000 })
    await expect(tablist.getByRole('tab', { name: 'Coverage' })).toHaveAttribute('aria-selected', 'true')

    await tablist.getByRole('tab', { name: 'Pipeline' }).click()
    await expect(tablist.getByRole('tab', { name: 'Pipeline' })).toHaveAttribute('aria-selected', 'true')
  })

  /**
   * #1619 category C — deliberately offline. `loadGraphViaFilePicker` aborts `/api/health` to
   * force the no-backend path, then loads the graph from a local file. A live API is the one
   * thing this test must NOT have.
   */
  test('offline graph load still reaches Dashboard briefing', async ({ page }) => {
    await loadGraphViaFilePicker(page)
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await expect(page.getByTestId('briefing-card')).toBeVisible()
  })

  test('Intelligence topic click opens Graph and topic detail rail', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()

    const tablist = page.getByRole('tablist', { name: 'Dashboard tabs' })
    await tablist.getByRole('tab', { name: 'Intelligence' }).click()
    await expect(page.getByTestId('intelligence-topic-landscape')).toBeVisible()

    await page
      .getByTestId('intelligence-topic-landscape')
      .locator('button[role="listitem"]')
      .first()
      .click()

    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
    // Intelligence cluster cards prefer `tc:…` compound id → graph node rail (NodeDetail / TopicCluster).
    await expect(page.getByTestId('graph-node-detail-rail')).toBeVisible({ timeout: 15_000 })
    await expect(page.getByTestId('graph-node-detail-rail')).toContainText(/TopicCluster/i)
  })

  /**
   * #1619 category B — still mocked, and it needs a v4 fixture, not a rewrite.
   *
   * Topic briefing cards render `digest.topics[]`, which is **retrieval-grounded**: the server
   * fills each topic's `hits[]` by running its configured digest query through the search index.
   * The v3 corpus configures no digest topic queries, so `/api/corpus/digest` returns
   * `topics: []` with `topics_unavailable_reason: null` — there is no assertion to rewrite,
   * because the surface under test has nothing to render. Recorded in
   * docs/wip/CORPUS-V4-FIXTURE-LADDER.md §B.
   */
  test('FR6.1: Intelligence shows retrieval-grounded topic briefing cards', async ({ page }) => {
    await setupDashboardApiMocks(page)
    // Override the digest mock with retrieval-scored topic bands.
    await page.route('**/api/corpus/digest**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          window: '7d',
          window_start_utc: '1970-01-01T00:00:00Z',
          window_end_utc: '2024-06-08T00:00:00Z',
          compact: false,
          rows: [],
          topics: [
            {
              topic_id: 't1',
              label: 'Climate Policy',
              query: 'climate policy',
              graph_topic_id: 'topic:climate',
              hits: [
                { metadata_relative_path: 'm/e1.json', episode_title: 'Ep A', feed_id: 'f1', score: 0.92, summary_preview: 'A grounded segment about climate policy.', publish_date: '2024-06-05', episode_id: 'e1' },
              ],
            },
          ],
          topics_unavailable_reason: null,
        }),
      })
    })
    await page.route('**/api/relational/cross-show**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          subject: 'topic:climate',
          groups: {
            'podcast:show-a': [
              { id: 'insight:1', type: 'insight', text: 'Show A take on climate.', show_id: 'podcast:show-a', episode_id: 'e1' },
            ],
          },
          error: null,
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await page.getByRole('tablist', { name: 'Dashboard tabs' }).getByRole('tab', { name: 'Intelligence' }).click()

    const cards = page.getByTestId('topic-briefing-cards')
    await expect(cards).toBeVisible({ timeout: 15_000 })
    const card = cards.getByTestId('topic-briefing-card').first()
    await expect(card).toContainText('Climate Policy')
    await expect(card).toContainText('0.92')
    await expect(card.getByTestId('topic-briefing-card-cross-show')).toContainText('Show A take on climate.')

    // Card link opens the Topic Entity View rail.
    await card.getByTestId('topic-briefing-card-link').click()
    await expect(page.getByTestId('topic-entity-view')).toBeVisible()
  })

  test('FR6.2: Intelligence shows the search-activity chart when there is data', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await page.getByRole('tablist', { name: 'Dashboard tabs' }).getByRole('tab', { name: 'Intelligence' }).click()

    /* The corpus ships `search/query_log.jsonl`, so the chart has real data. Assert that it
     * reports a positive count — NOT an exact one.
     *
     * The query log is APPEND-ONLY and live: every search any other spec runs during the same
     * suite adds to it. Pinning the number read a moment earlier made this flaky, because a
     * concurrent worker's search moved it between the read and the render. */
    const resp = await page.request.get('/api/corpus/query-activity')
    const { total } = (await resp.json()) as { total: number }
    expect(total).toBeGreaterThan(0)

    const chart = page.getByTestId('query-activity-chart')
    await expect(chart).toBeVisible({ timeout: 15_000 })
    await expect(chart).toContainText('Search activity')
    await expect(chart).toContainText(/\d+ searches/)
  })
})
