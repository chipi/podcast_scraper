import { expect, test } from '@playwright/test'
import { setupDashboardApiMocks } from './dashboardApiMocks'
import {
  loadGraphViaFilePicker,
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

async function mockDashboardApis(page: import('@playwright/test').Page): Promise<void> {
  await setupDashboardApiMocks(page)
  await page.route('**/api/index/stats**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        available: true,
        index_path: '/mock/corpus/.idx',
        stats: {
          total_vectors: 100,
          doc_type_counts: { episode: 100 },
          feeds_indexed: ['f1', 'f2'],
          embedding_model: 'mock-model',
          embedding_dim: 4,
          last_updated: '2024-01-01T12:00:00Z',
          index_size_bytes: 4096,
        },
        reindex_recommended: false,
        reindex_reasons: [],
        artifact_newest_mtime: '2024-01-01T14:00:00Z',
        search_root_hints: [],
        rebuild_in_progress: false,
        rebuild_last_error: null,
      }),
    })
  })
}

test.describe('Dashboard tab', () => {
  test('briefing shows no-corpus empty state when path is unset', async ({ page }) => {
    await mockDashboardApis(page)
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

  test('shows briefing card after opening Dashboard (mocked API)', async ({ page }) => {
    await mockDashboardApis(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await expect(page.getByTestId('briefing-card')).toBeVisible({ timeout: 15_000 })
  })

  test('Coverage tab is default; Pipeline tab can be selected', async ({ page }) => {
    await mockDashboardApis(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()

    const tablist = page.getByRole('tablist', { name: 'Dashboard tabs' })
    await expect(tablist).toBeVisible({ timeout: 15_000 })
    await expect(tablist.getByRole('tab', { name: 'Coverage' })).toHaveAttribute('aria-selected', 'true')

    await tablist.getByRole('tab', { name: 'Pipeline' }).click()
    await expect(tablist.getByRole('tab', { name: 'Pipeline' })).toHaveAttribute('aria-selected', 'true')
  })

  test('offline graph load still reaches Dashboard briefing', async ({ page }) => {
    await loadGraphViaFilePicker(page)
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await expect(page.getByTestId('briefing-card')).toBeVisible()
  })
})
