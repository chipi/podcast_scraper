import { expect, test } from '@playwright/test'
import { leftPanelTabs, SHELL_HEADING_RE } from './helpers'

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

test.describe('Index rebuild from API · Data panel (mocked API)', () => {
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
  })

  test('Update index sends POST /api/index/rebuild (202, incremental)', async ({ page }) => {
    await page.route('**/api/index/rebuild**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.continue()
        return
      }
      await route.fulfill({
        status: 202,
        contentType: 'application/json',
        body: JSON.stringify({
          accepted: true,
          corpus_path: '/mock/corpus',
          rebuild: false,
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await leftPanelTabs(page).getByRole('button', { name: 'API · Data' }).click()
    await expect(page.getByRole('heading', { name: 'Vector index' })).toBeVisible()

    const updateBtn = page.getByRole('button', { name: 'Update index' })
    await expect(updateBtn).toBeEnabled()

    const reqPromise = page.waitForRequest(
      (req) => req.url().includes('/api/index/rebuild') && req.method() === 'POST',
    )
    await updateBtn.click()
    const req = await reqPromise
    const u = new URL(req.url())
    expect(u.searchParams.get('rebuild')).not.toBe('true')
  })

  test('Full rebuild sends POST with rebuild=true', async ({ page }) => {
    await page.route('**/api/index/rebuild**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.continue()
        return
      }
      await route.fulfill({
        status: 202,
        contentType: 'application/json',
        body: JSON.stringify({
          accepted: true,
          corpus_path: '/mock/corpus',
          rebuild: true,
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await leftPanelTabs(page).getByRole('button', { name: 'API · Data' }).click()
    await expect(page.getByRole('heading', { name: 'Vector index' })).toBeVisible()

    const fullBtn = page.getByRole('button', { name: 'Full rebuild' })
    await expect(fullBtn).toBeEnabled()

    const reqPromise = page.waitForRequest(
      (req) => req.url().includes('/api/index/rebuild') && req.method() === 'POST',
    )
    await fullBtn.click()
    const req = await reqPromise
    expect(new URL(req.url()).searchParams.get('rebuild')).toBe('true')
  })
})
