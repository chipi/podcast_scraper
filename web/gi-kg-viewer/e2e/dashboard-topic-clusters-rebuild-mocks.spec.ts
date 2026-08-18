import { expect, test } from '@playwright/test'
import { setupCorpusDashboardDataRoutes } from './dashboardApiMocks'
import { openCorpusDataWorkspace, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * task-#14: the Dashboard "Topic clusters" card exposes a Rebuild button when clusters are missing,
 * so operators regenerate search/topic_clusters.json without CLI/SSH. This drives that button
 * end-to-end in the browser against a mocked API: GET returns 404 (missing) → the button shows →
 * clicking fires POST /api/corpus/topic-clusters/rebuild → the poll then sees clusters and the card
 * flips to Loaded.
 *
 * #1619 category C — permanently mocked, and correctly so.
 *
 * Live, the click triggers a **real** topic-cluster rebuild: an embedding pass that regenerates
 * `search/topic_clusters.json` inside the corpus. An e2e test must not do that — slow, needs the
 * ML stack, and overwrites a committed fixture artifact. The starting state is unreachable too:
 * the test needs clusters *missing* (GET → 404) and the v3 corpus ships them. The rebuild itself
 * belongs to the Python tests.
 */
test.describe('Topic-clusters rebuild button (mocked API)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin')
    await page.route('**/api/health**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok', corpus_library_api: true, corpus_digest_api: true }),
      })
    })
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
      })
    })
    await setupCorpusDashboardDataRoutes(page)
  })

  test('missing → Rebuild fires POST and the card flips to Loaded', async ({ page }) => {
    let rebuilt = false

    // topic-clusters GET: 404 until a rebuild is POSTed, then a valid (non-empty) document.
    await page.route('**/api/corpus/topic-clusters?**', async (route) => {
      if (rebuilt) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ schema_version: '2', clusters: [{ graph_compound_parent_id: 'tc:x' }] }),
        })
      } else {
        await route.fulfill({
          status: 404,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'not found', available: false }),
        })
      }
    })
    await page.route('**/api/corpus/topic-clusters/rebuild**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.continue()
        return
      }
      rebuilt = true
      await route.fulfill({
        status: 202,
        contentType: 'application/json',
        body: JSON.stringify({ accepted: true, corpus_path: '/mock/corpus' }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await openCorpusDataWorkspace(page)
    // Intelligence sub-tab hosts the Topic clusters card. Crucially we do NOT open the Graph tab
    // first — the card self-fetches its status, so the Rebuild button surfaces on the dashboard
    // without a graph interaction (the fix for the App.vue graph-gated sync).
    await page
      .getByRole('tablist', { name: 'Dashboard tabs' })
      .getByRole('tab', { name: 'Intelligence' })
      .click()

    const rebuildBtn = page.getByTestId('topic-clusters-rebuild')
    await expect(rebuildBtn).toBeVisible() // shown because clusters are missing (404)

    const reqPromise = page.waitForRequest(
      (req) =>
        req.url().includes('/api/corpus/topic-clusters/rebuild') && req.method() === 'POST',
    )
    await rebuildBtn.click()
    await reqPromise // the button → POST wiring fired

    // After the rebuild + poll, the status card reflects Loaded and the Rebuild button is gone.
    const card = page.getByTestId('topic-clusters-status-block')
    await expect(card).toContainText('Loaded', { timeout: 10_000 })
    await expect(page.getByTestId('topic-clusters-rebuild')).toHaveCount(0)
  })
})
