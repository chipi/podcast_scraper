import { expect, test, type Page } from '@playwright/test'
import { setupDashboardApiMocks } from './dashboardApiMocks'
import { mainViewsNav, mockSignIn, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

/**
 * E04: without the insight stage the server writes and serves KG only — no ``.gi.json``, no
 * bridge, and no CIL / explore / top-persons routes. That is a normal corpus, not a broken one:
 * the viewer must not call the absent routes, nor warn that GI is missing.
 *
 * Mocked on purpose: the v3 fixture corpus has GI for every episode, and the live server mounts
 * the insight stage's routers, so neither can be put in this state without a second build.
 */

/** Routes only the insight stage mounts. Each answers 404 here, as on a KG-only server. */
const STAGE_ROUTES = [
  '**/api/topics/**',
  '**/api/persons/**',
  '**/api/corpus/persons/top**',
  '**/api/explore**',
]

async function fulfillJson(page: Page, pattern: string, body: unknown): Promise<void> {
  await page.route(pattern, (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) }),
  )
}

async function setupKgOnlyServer(page: Page): Promise<string[]> {
  const stageCalls: string[] = []
  await mockSignIn(page, 'admin')
  await setupDashboardApiMocks(page)
  for (const pattern of STAGE_ROUTES) {
    await page.route(pattern, (route) => {
      stageCalls.push(route.request().url())
      return route.fulfill({ status: 404, contentType: 'application/json', body: '{"detail":"Not Found"}' })
    })
  }
  // Registered after the shared mocks, so these win (Playwright matches the newest route first).
  await fulfillJson(page, '**/api/health**', {
    status: 'ok',
    corpus_library_api: true,
    corpus_digest_api: true,
    explore_api: false,
    cil_queries_api: false,
    corpus_persons_api: false,
    artifact_kinds: ['kg'],
  })
  await fulfillJson(page, '**/api/artifacts?**', {
    path: '/mock/corpus',
    artifacts: [
      {
        name: 'ep.kg.json',
        relative_path: 'feeds/f/run_1/metadata/ep.kg.json',
        kind: 'kg',
        size_bytes: 10,
        mtime_utc: '2026-11-01T12:00:00Z',
        publish_date: '2026-11-01',
      },
    ],
  })
  await fulfillJson(page, '**/api/corpus/coverage?**', {
    path: '/mock/corpus',
    total_episodes: 10,
    with_gi: 0,
    with_kg: 10,
    with_both: 0,
    with_neither: 0,
    by_month: [],
    by_feed: [{ feed_id: 'f', display_title: 'Feed', total: 10, with_gi: 0, with_kg: 10 }],
  })
  return stageCalls
}

test.describe('KG-only corpus (no insight stage)', () => {
  test('the dashboard renders without GI warnings and never calls the insight routes', async ({
    page,
  }) => {
    const stageCalls = await setupKgOnlyServer(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()

    await expect(page.getByTestId('briefing-card')).toBeVisible()
    await expect(page.getByText(/have no GI artifacts/)).toHaveCount(0)
    await expect(page.getByText(/lowest GI coverage/)).toHaveCount(0)

    // The multi-perspective panel lives on the Intelligence tab and reads only CIL routes.
    await page
      .getByRole('tablist', { name: 'Dashboard tabs' })
      .getByRole('tab', { name: 'Intelligence' })
      .click()
    await expect(page.getByTestId('dashboard-tab-intelligence')).toHaveAttribute(
      'aria-selected',
      'true',
    )
    await expect(page.getByTestId('dashboard-topic-perspectives')).toHaveCount(0)
    // Give every dashboard panel its load cycle before reading the call log.
    await page.waitForLoadState('networkidle')
    expect(stageCalls).toEqual([])
  })
})
