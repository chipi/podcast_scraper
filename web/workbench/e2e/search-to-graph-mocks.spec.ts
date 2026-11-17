import { readFileSync } from 'node:fs'
import { expect, test } from '@playwright/test'
import { KG_SAMPLE_FIXTURE } from './fixtures'
import { setupCorpusDashboardDataRoutes } from './dashboardApiMocks'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

const artifactJson = readFileSync(KG_SAMPLE_FIXTURE, 'utf-8')

/**
 * #1619 — stays fixture-driven, deliberately. It serves **`KG_SAMPLE_FIXTURE`** — a checked-in
 * graph artifact — through `/api/artifacts/metadata/ci_sample.kg.json` and drives search against
 * that graph, so the assertions do not move with whatever today's ranking surfaces. The Insight's
 * graph detail (provenance, supporting quotes, transcript) is a downstream build's, with GI, and so
 * are the result card's supporting quotes and lifted block (E04 step 8b): their tests moved there.
 */
test.describe('Search → graph (mocked API)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin')
  })

  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          artifact_kinds: ['kg'],
        }),
      })
    })

    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          artifacts: [
            {
              name: 'ci_sample.kg.json',
              relative_path: 'metadata/ci_sample.kg.json',
              kind: 'kg',
              size_bytes: artifactJson.length,
              mtime_utc: '2026-04-18T12:00:00Z',
              publish_date: '2026-04-18',
            },
          ],
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/ci_sample.kg.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: artifactJson,
      })
    })

    await page.route('**/api/corpus/feeds**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', feeds: [] }),
      })
    })

    await page.route('**/api/corpus/episodes/detail**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          metadata_relative_path: 'metadata/ci_sample.metadata.json',
          feed_id: 'stub-feed',
          episode_id: 'ci-fixture',
          episode_title: 'CI fixture episode',
          publish_date: '2020-01-01',
          summary_title: 'Stub',
          summary_bullets: [],
          summary_text: null,
          gi_relative_path: null,
          kg_relative_path: 'metadata/ci_sample.kg.json',
          has_gi: false,
          has_kg: true,
        }),
      })
    })

    await page.route('**/api/index/stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          available: false,
          reason: 'mock-off',
          stats: null,
          reindex_recommended: false,
        }),
      })
    })

    await page.route('**/api/corpus/episodes/similar**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          source_metadata_relative_path: 'metadata/ci_sample.metadata.json',
          query_used: '',
          items: [],
          error: null,
          detail: null,
        }),
      })
    })

    await page.route('**/api/search?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: 'stub-query',
          results: [
            {
              doc_id: 'doc-1',
              score: 0.95,
              text: 'Climate policy topic (stub).',
              metadata: {
                doc_type: 'kg_topic',
                source_id: 'topic:ci-policy',
                episode_id: 'ci-fixture',
                source_metadata_relative_path: 'metadata/ci_sample.metadata.json',
              },
            },
          ],
        }),
      })
    })
    await setupCorpusDashboardDataRoutes(page)
  })

  test('corpus path auto-loads graph → search → Show on graph opens node detail', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await page.locator('#search-q').fill('climate insights')
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()

    await page.getByTestId('search-workspace').getByText('Climate policy topic (stub)', { exact: false }).first().waitFor({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Search result insights' }).click()
    const vizDialog = page.getByRole('dialog', { name: 'Search result insights' })
    await expect(vizDialog.getByRole('region', { name: 'Doc types' })).toBeVisible()
    await expect(vizDialog.getByRole('region', { name: 'Publish month' })).toBeVisible()
    await vizDialog.getByRole('button', { name: 'Close' }).click()

    await page.getByRole('button', { name: 'Show on graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
    await expect(page.locator('.graph-canvas')).toBeVisible()
  })
})
