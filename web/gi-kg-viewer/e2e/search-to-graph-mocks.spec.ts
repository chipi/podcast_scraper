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
 * graph detail (provenance, supporting quotes, transcript) is a downstream build's, with GI.
 *
 * The genuinely corpus-blocked parts, measured rather than assumed: the **`lifted` quote tests**
 * (`search supporting quotes without speaker…`, `transcript hit with lifted quote timing…`) — the
 * corpus returns `"lifted": null` on every result (50/50 across five queries), so the
 * muted-attribution hints have no reachable state. Ladder §B.
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

  test('search supporting quotes without speaker show muted attribution hint', async ({ page }) => {
    await page.unroute('**/api/search?**')
    await page.route('**/api/search?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: 'with-quotes',
          results: [
            {
              doc_id: 'doc-ins-q',
              score: 0.92,
              text: 'Insight with supporting quote (no speaker).',
              metadata: {
                doc_type: 'insight',
                source_id: 'insight:with-q',
                episode_id: 'ci-fixture',
                source_metadata_relative_path: 'metadata/ci_sample.metadata.json',
              },
              supporting_quotes: [{ text: 'Quoted evidence without speaker fields.' }],
            },
          ],
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await page.locator('#search-q').fill('with quotes')
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()

    await page.getByText('Insight with supporting quote', { exact: false }).waitFor({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Show 1 supporting quote' }).click()

    const hint = page.getByTestId('supporting-quote-speaker-unavailable')
    await expect(hint).toBeVisible()
    await expect(hint).toContainText('No speaker detected')
  })

  test('transcript hit with lifted quote timing and no speaker shows muted lifted hint', async ({
    page,
  }) => {
    await page.unroute('**/api/search?**')
    await page.route('**/api/search?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: 'lift-no-speaker',
          lift_stats: { transcript_hits_returned: 1, lift_applied: 1 },
          results: [
            {
              doc_id: 'doc-tr-lift',
              score: 0.88,
              text: 'Transcript chunk text for lift hint.',
              metadata: {
                doc_type: 'transcript',
                episode_id: 'ci-fixture',
                source_metadata_relative_path: 'metadata/ci_sample.metadata.json',
                char_start: 0,
                char_end: 40,
              },
              lifted: {
                insight: {
                  id: 'ins:lift-hint',
                  text: 'Linked GI insight (lift mock).',
                  grounded: true,
                },
                quote: { timestamp_start_ms: 5000, timestamp_end_ms: 9000 },
              },
            },
          ],
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await page.locator('#search-q').fill('lift no speaker')
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()

    await page.getByText('Transcript chunk text for lift hint', { exact: false }).waitFor({
      timeout: 10_000,
    })

    // Lifted block defaults expanded (`liftedOpen`); toggle label is **Hide** linked GI insight.
    const liftedHint = page.getByTestId('search-lifted-quote-speaker-unavailable')
    await expect(liftedHint).toBeVisible()
    await expect(liftedHint).toContainText('No speaker detected')
  })
})
