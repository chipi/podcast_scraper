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

  test('Intelligence topic click opens Graph and topic detail rail', async ({ page }) => {
    await mockDashboardApis(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
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

  test('FR6.1: Intelligence shows retrieval-grounded topic briefing cards', async ({ page }) => {
    await mockDashboardApis(page)
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
})
