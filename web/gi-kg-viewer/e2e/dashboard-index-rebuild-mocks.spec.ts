import { expect, test } from '@playwright/test'
import { setupCorpusDashboardDataRoutes } from './dashboardApiMocks'
import { openCorpusDataWorkspace, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/** Minimal `GET /api/index/stats` body so Dashboard enables index actions (#507). */
const INDEX_STATS_ENVELOPE = {
  available: true,
  index_path: '/mock/corpus/search',
  stats: {
    total_vectors: 1,
    doc_type_counts: { insight: 1 },
    feeds_indexed: [],
    embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
    embedding_dim: 384,
    last_updated: '2024-01-01T00:00:00Z',
    index_size_bytes: 100,
  },
  reindex_recommended: false,
  reindex_reasons: [] as string[],
  artifact_newest_mtime: null as string | null,
  search_root_hints: [] as string[],
  rebuild_in_progress: false,
  rebuild_last_error: null as string | null,
}

/**
 * Config consolidation: the Dashboard **Index status** card is status-only. The
 * rebuild *action* lives in the Configuration "Vector index" dialog (single
 * canonical place per concern). The card's **Manage in Configuration** button
 * routes there; the dialog then issues `POST /api/index/rebuild`.
 */
test.describe('Index rebuild via Configuration Vector index dialog (mocked API)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin')
  })

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
    await page.route('**/api/index/stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(INDEX_STATS_ENVELOPE),
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

  test('Index status card is status-only and routes rebuild to Configuration (incremental)', async ({
    page,
  }) => {
    await page.route('**/api/index/rebuild**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.continue()
        return
      }
      await route.fulfill({
        status: 202,
        contentType: 'application/json',
        body: JSON.stringify({ accepted: true, corpus_path: '/mock/corpus', rebuild: false }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await openCorpusDataWorkspace(page)

    const card = page.getByTestId('index-status-card')
    await expect(card).toBeVisible()
    // The inline rebuild buttons have moved out of the dashboard card.
    await expect(page.getByTestId('index-status-update')).toHaveCount(0)
    await expect(page.getByTestId('index-status-full-rebuild')).toHaveCount(0)

    // Manage opens the Configuration dialog at its Index section (the rebuild home).
    await card.getByTestId('index-status-manage').click()
    await expect(page.getByTestId('status-bar-sources-dialog')).toBeVisible()
    await expect(page.getByTestId('sources-dialog-index-panel')).toBeVisible()

    const reqPromise = page.waitForRequest(
      (req) => req.url().includes('/api/index/rebuild') && req.method() === 'POST',
    )
    await page.getByTestId('index-dialog-update').click()
    const req = await reqPromise
    expect(new URL(req.url()).searchParams.get('rebuild')).not.toBe('true')
  })

  test('Full rebuild from the Configuration dialog sends rebuild=true', async ({ page }) => {
    await page.route('**/api/index/rebuild**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.continue()
        return
      }
      await route.fulfill({
        status: 202,
        contentType: 'application/json',
        body: JSON.stringify({ accepted: true, corpus_path: '/mock/corpus', rebuild: true }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await openCorpusDataWorkspace(page)

    await page.getByTestId('index-status-card').getByTestId('index-status-manage').click()
    await expect(page.getByTestId('status-bar-sources-dialog')).toBeVisible()
    await expect(page.getByTestId('sources-dialog-index-panel')).toBeVisible()

    const reqPromise = page.waitForRequest(
      (req) => req.url().includes('/api/index/rebuild') && req.method() === 'POST',
    )
    await page.getByTestId('index-dialog-full-rebuild').click()
    const req = await reqPromise
    expect(new URL(req.url()).searchParams.get('rebuild')).toBe('true')
  })
})
