import { readFileSync } from 'node:fs'
import { expect, test } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from './fixtures'
import { leftPanelTabs, mainViewsNav, SHELL_HEADING_RE } from './helpers'

const artifactJson = readFileSync(GI_SAMPLE_FIXTURE, 'utf-8')

const TRANSCRIPT_BODY =
  'Hello world transcript sample for CI quality metrics fixture.\nExtra line for scroll.'

test.describe('Search → graph (mocked API)', () => {
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

    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          artifacts: [
            {
              name: 'ci_sample.gi.json',
              relative_path: 'metadata/ci_sample.gi.json',
              kind: 'gi',
              size_bytes: artifactJson.length,
              mtime_utc: '2024-01-01T00:00:00Z',
            },
          ],
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/ci_sample.gi.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: artifactJson,
      })
    })

    await page.route('**/api/corpus/text-file**', async (route) => {
      const url = new URL(route.request().url())
      const relpath = decodeURIComponent(url.searchParams.get('relpath') || '')
      if (relpath.endsWith('.segments.json')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json; charset=utf-8',
          body: JSON.stringify([]),
        })
        return
      }
      if (relpath.endsWith('transcript.txt')) {
        await route.fulfill({
          status: 200,
          contentType: 'text/plain; charset=utf-8',
          body: TRANSCRIPT_BODY,
        })
        return
      }
      await route.fulfill({ status: 404, body: 'not found' })
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
          gi_relative_path: 'metadata/ci_sample.gi.json',
          kg_relative_path: 'metadata/ci_sample.kg.json',
          has_gi: true,
          has_kg: false,
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
              text: 'Summary insight (stub).',
              metadata: {
                doc_type: 'insight',
                source_id: 'insight:b72dafa3f874480d',
                episode_id: 'ci-fixture',
                source_metadata_relative_path: 'metadata/ci_sample.metadata.json',
              },
            },
          ],
        }),
      })
    })

    await page.route('**/api/corpus/topic-clusters**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          schema_version: '2',
          clusters: [
            {
              graph_compound_parent_id: 'tc:ci-policy-cluster',
              cil_alias_target_topic_id: 'topic:ci-policy',
              canonical_label: 'Climate policy cluster',
              member_count: 1,
              members: [{ topic_id: 'topic:ci-policy' }],
            },
          ],
          topic_count: 1,
          cluster_count: 1,
          singletons: 0,
        }),
      })
    })
  })

  test('RFC-075: mocked topic-clusters v2 adds Topic cluster to Types row', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await expect(page.getByText(/Topic cluster\s*\(\d+\)/)).toBeVisible()
  })

  test('RFC-075: API Data tab shows topic clusters loaded and schema line', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await leftPanelTabs(page).getByRole('button', { name: 'API · Data' }).click()
    const apiSection = page.locator('section').filter({
      has: page.getByRole('heading', { name: 'API', exact: true }),
    })
    await expect(
      apiSection.getByRole('heading', { name: 'Topic clusters', level: 3 }),
    ).toBeVisible()
    await expect(apiSection.getByText('Loaded', { exact: true })).toBeVisible()
    await expect(apiSection.getByText(/schema_version:\s*2/)).toBeVisible()
  })

  test('corpus path auto-loads graph → search → Show on graph opens node detail', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await page.locator('#search-q').fill('climate insights')
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()

    await page.getByText('Summary insight (stub)', { exact: false }).waitFor({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Search result insights' }).click()
    const vizDialog = page.getByRole('dialog', { name: 'Search result insights' })
    await expect(vizDialog.getByRole('region', { name: 'Doc types' })).toBeVisible()
    await expect(vizDialog.getByRole('region', { name: 'Publish month' })).toBeVisible()
    await vizDialog.getByRole('button', { name: 'Close' }).click()

    await page.getByRole('button', { name: 'Show on graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
    await expect(page.locator('.graph-canvas')).toBeVisible()
  })

  test('insight graph detail: provenance, related topics, Library, quotes, Explore, Search', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await page.locator('#search-q').fill('climate insights')
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()

    await page.getByText('Summary insight (stub)', { exact: false }).waitFor({ timeout: 10_000 })
    await page.getByRole('button', { name: 'Show on graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await expect(page.getByRole('region', { name: 'Graph node: Insight' })).toBeVisible({
      timeout: 15_000,
    })

    await page
      .getByTestId('node-detail-insight-details-tip')
      .getByRole('button', { name: 'Grounded' })
      .click()
    await expect(page.getByTestId('node-detail-insight-details-tooltip-body')).toContainText(
      'Grounded',
    )
    await expect(page.getByTestId('node-detail-insight-details-tooltip-body')).toContainText('stub')
    await expect(page.getByTestId('node-detail-insight-details-tooltip-body')).toContainText(
      'ci_sample',
    )
    await expect(page.getByTestId('node-detail-insight-related-topics')).toBeVisible()
    await expect(page.getByTestId('node-detail-insight-related-topics')).toContainText(
      'Climate policy',
    )
    await expect(page.getByTestId('node-detail-insight-supporting-quotes')).toBeVisible()
    await expect(page.getByTestId('node-detail-insight-supporting-quotes')).toContainText(
      'Hello world transcript sample',
    )
    await expect(page.getByTestId('node-detail-insight-supporting-quotes')).toContainText(
      'Extra line for scroll',
    )
    await page.getByTestId('node-detail-insight-view-transcript-all-quotes').click()
    const transcriptDlg = page.getByTestId('transcript-viewer-dialog')
    await expect(transcriptDlg).toBeVisible()
    await expect(transcriptDlg.getByTestId('transcript-viewer-char-range')).toContainText(
      '2 character spans (supporting quotes)',
    )
    await expect(transcriptDlg.getByTestId('transcript-viewer-highlight')).toHaveCount(2)
    await expect(transcriptDlg.getByTestId('transcript-viewer-body')).toContainText('Hello world')
    await expect(transcriptDlg.getByTestId('transcript-viewer-body')).toContainText('Extra line')
    await transcriptDlg.getByRole('button', { name: 'Close' }).click()
    await expect(transcriptDlg).toBeHidden()

    const episodeConnRow = page.locator('[data-connection-node-id="episode:ci-fixture"]')
    await expect(episodeConnRow.getByTestId('graph-connection-open-library')).toBeEnabled()
    await expect(episodeConnRow.getByTestId('graph-connection-focus-graph')).toBeEnabled()
    await expect(episodeConnRow.getByTestId('graph-connection-prefill-search')).toBeEnabled()

    const insightTipBody = page.getByTestId('node-detail-insight-details-tooltip-body')
    if (await insightTipBody.isVisible()) {
      await page
        .getByTestId('node-detail-insight-details-tip')
        .getByRole('button', { name: 'Grounded' })
        .click()
    }
    await expect(page.getByRole('tooltip')).toHaveCount(0)

    await page.getByTestId('node-detail-insight-explore-filters').click()
    await expect(
      page.getByRole('heading', { name: 'Explore & query', exact: false }),
    ).toBeVisible({ timeout: 10_000 })
    await expect(page.getByRole('checkbox', { name: /Grounded only/i })).toBeChecked()
    await expect(page.getByRole('textbox', { name: 'Topic contains' })).toHaveValue('')
    await expect(page.getByRole('textbox', { name: 'Speaker contains' })).toHaveValue('')

    await page.getByRole('button', { name: 'Back to details' }).click()
    await expect(page.getByTestId('node-detail-insight-details-tip')).toBeVisible()

    await page.getByTestId('node-detail-insight-prefill-search').click()
    await expect(page.locator('#search-q')).toHaveValue(/Summary insight \(stub\)/)

    await page.getByRole('button', { name: 'Back to details' }).click()
    await expect(page.getByTestId('node-detail-insight-related-topics')).toBeVisible()

    await page
      .getByTestId('node-detail-insight-related-topic-row')
      .filter({ hasText: 'Climate policy' })
      .click()
    await expect(page.getByRole('region', { name: 'Graph node: Topic' })).toBeVisible({
      timeout: 10_000,
    })
    await expect(page.getByTestId('node-detail-full-topic')).toContainText('Climate policy')

    await page.getByRole('button', { name: 'Search & Explore' }).click()
    await page.locator('#search-q').fill('climate insights')
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()
    await page.getByText('Summary insight (stub)', { exact: false }).waitFor({ timeout: 10_000 })
    await page.getByRole('button', { name: 'Show on graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
    await expect(page.getByRole('region', { name: 'Graph node: Insight' })).toBeVisible({
      timeout: 15_000,
    })
    const episodeRowAgain = page.locator('[data-connection-node-id="episode:ci-fixture"]')
    await expect(episodeRowAgain.getByTestId('graph-connection-open-library')).toBeEnabled()

    await episodeRowAgain.getByTestId('graph-connection-open-library').click()
    await expect(mainViewsNav(page).getByRole('button', { name: 'Library' })).toHaveClass(
      /bg-primary/,
    )
    await expect(page.getByTestId('episode-detail-rail-body')).toContainText('CI fixture episode', {
      timeout: 15_000,
    })
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

    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

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

    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

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
