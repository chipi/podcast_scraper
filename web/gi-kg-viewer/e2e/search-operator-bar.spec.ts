import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * Search v3 §S4 (#1234) — ResultSetOperatorBar contract on the Search main tab.
 *
 * Covers S4a (client-only Timeline + On-graph). The server-side Cluster and Consensus operators
 * are private features (ADR-162) and never appear in this viewer.
 *
 * The E2E surface map — [E2E_SURFACE_MAP.md](E2E_SURFACE_MAP.md) — is the canonical selector
 * contract; the testids referenced here are documented in the "Result-set operator bar (#1234)"
 * block of that file.
 *
 * #1619 — runs against the live index. Where the corpus is thinner than the old fixture, the
 * assertion says so instead of pretending: every live result carries a `publish_date` (so the
 * Timeline "undated" tally is 0, not 1). Recorded in docs/architecture/TEST_CORPUS_FIXTURE_LADDER.md
 * §B.
 */
const QUERY = 'systems thinking'

test.describe('Search — result-set operator bar (#1234)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  /** Land the Search tab, submit a real query, wait for the operator bar. */
  async function runSearchAndWaitForBar(page: Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await page.locator('#search-q').fill(QUERY)
    // Enter submits (SearchPanel's on-keydown handler); avoids the form-linked
    // Search button-scope pattern that broke in an earlier iteration.
    await page.locator('#search-q').press('Enter')
    await expect(page.getByTestId('result-set-operator-bar')).toBeVisible({ timeout: 30_000 })
  }

  test('renders the Timeline / On graph / Compare chips and no private operator', async ({
    page,
  }) => {
    await runSearchAndWaitForBar(page)
    await expect(page.getByTestId('operator-chip-timeline')).toBeVisible()
    await expect(page.getByTestId('operator-chip-compare')).toBeVisible()
    await expect(page.getByTestId('operator-chip-cluster')).toHaveCount(0)
    await expect(page.getByTestId('operator-chip-consensus')).toHaveCount(0)
    /* The On-graph chip label carries the count of graph-resolvable ids in the result set. That
     * count is ranking-dependent, so assert the shape and that it resolved something — pinning a
     * number would just restate whatever the corpus ranked today. */
    await expect(page.getByTestId('operator-chip-graph')).toHaveText(/On graph \(\d+\)/)
    const graphLabel = await page.getByTestId('operator-chip-graph').textContent()
    expect(Number(/\((\d+)\)/.exec(graphLabel ?? '')?.[1] ?? '0')).toBeGreaterThan(0)
  })

  test('Timeline: toggles the dot chart on / off', async ({ page }) => {
    await runSearchAndWaitForBar(page)
    await expect(page.getByTestId('operator-timeline-panel')).toHaveCount(0)
    await page.getByTestId('operator-chip-timeline').click()
    const panel = page.getByTestId('operator-timeline-panel')
    await expect(panel).toBeVisible()
    await expect(page.getByTestId('operator-chip-timeline')).toHaveAttribute('aria-pressed', 'true')

    /* The old fixture included one hit with no `publish_date` so the "undated" notice rendered.
     * Every result the live corpus returns is dated, so that branch is unreachable here — assert
     * it is absent rather than deleting the coverage silently. A v4 corpus with an undated
     * artifact would flip this.
     *
     * Asserted from the DOM alone, deliberately: confirming it via a second `/api/search` cost an
     * extra query embedding on a single-worker backend for a fact the panel already shows. */
    await expect(page.getByTestId('operator-timeline-undated')).toHaveCount(0)

    // Second click toggles the panel off; chip returns to unpressed.
    await page.getByTestId('operator-chip-timeline').click()
    await expect(panel).toHaveCount(0)
    await expect(page.getByTestId('operator-chip-timeline')).toHaveAttribute(
      'aria-pressed',
      'false',
    )
  })

  test('On graph: pressing the chip switches to the Graph main tab', async ({ page }) => {
    await runSearchAndWaitForBar(page)
    // Sanity: the Search workspace is visible before the handoff.
    await expect(page.getByTestId('search-workspace')).toBeVisible()
    await page.getByTestId('operator-chip-graph').click()
    // The Search workspace unmounts once ``mainTab === 'graph'`` — that's the load-bearing
    // observable that App.vue's ``activateGraphTab('search')`` fired.
    await expect(page.getByTestId('search-workspace')).toHaveCount(0, { timeout: 10_000 })
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
  })
})
