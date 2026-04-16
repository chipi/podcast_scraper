import { readFileSync } from 'node:fs'
import { expect, test, type Page } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from './fixtures'
import { mainViewsNav, SHELL_HEADING_RE } from './helpers'

const artifactJson = readFileSync(GI_SAMPLE_FIXTURE, 'utf-8')

/**
 * Mocks aligned with ``search-to-graph-mocks.spec.ts`` plus RFC-075
 * ``members[].episode_ids`` so ``maybeMergeClusterSiblingEpisodes`` runs after graph load.
 */
async function mockCorpusGraphBaseline(page: Page): Promise<void> {
  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        corpus_library_api: true,
        corpus_digest_api: true,
        cil_queries_api: true,
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

  await page.route('**/api/corpus/feeds**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ path: '/mock/corpus', feeds: [] }),
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

  await page.route('**/api/corpus/topic-clusters**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        schema_version: '2',
        clusters: [
          {
            graph_compound_parent_id: 'tc:ci-policy-cluster',
            canonical_label: 'Climate policy cluster',
            members: [
              {
                topic_id: 'topic:ci-policy',
                episode_ids: ['ci-fixture', 'ep-sibling-mock'],
              },
            ],
          },
        ],
        topic_count: 1,
        cluster_count: 1,
        singletons: 0,
      }),
    })
  })
}

test.describe('Topic-cluster sibling merge (mocked API)', () => {
  test('resolve failure shows alert banner and dismiss clears it', async ({ page }) => {
    await mockCorpusGraphBaseline(page)

    await page.route('**/api/corpus/resolve-episode-artifacts**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      await route.fulfill({
        status: 500,
        contentType: 'text/plain; charset=utf-8',
        body: 'resolve failed e2e',
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    const banner = page.getByTestId('sibling-merge-error-banner')
    await expect(banner).toBeVisible({ timeout: 30_000 })
    await expect(banner).toContainText('resolve failed e2e')

    await page.getByTestId('sibling-merge-error-dismiss').click()
    await expect(banner).toBeHidden()
  })

  test('empty resolve shows compact sibling line on Graph tab', async ({ page }) => {
    await mockCorpusGraphBaseline(page)

    await page.route('**/api/corpus/resolve-episode-artifacts**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          resolved: [],
          missing_episode_ids: ['ep-sibling-mock'],
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    const line = page.getByTestId('graph-sibling-merge-line')
    await expect(line).toBeVisible({ timeout: 30_000 })
    await expect(line).toHaveText(/^\+\d+ new · \d+ in cluster · cap \d+( · \d+ miss(es)?)?$/)
  })
})
