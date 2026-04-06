import { readFileSync } from 'node:fs'
import { expect, test } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from './fixtures'

const artifactJson = readFileSync(GI_SAMPLE_FIXTURE, 'utf-8')

test.describe('Search → graph (mocked API)', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok' }),
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
              },
            },
          ],
        }),
      })
    })
  })

  test('list → load → search → Show on graph opens node detail', async ({ page }) => {
    await page.goto('/')

    await page.getByLabel('Corpus root folder').fill('/mock/corpus')
    await page.getByRole('button', { name: 'List files' }).click()

    await page.getByRole('checkbox', { name: /ci_sample\.gi\.json/ }).check()

    await page.getByRole('button', { name: 'Load selected into graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await page.locator('#search-q').fill('climate insights')
    await page.getByRole('button', { name: 'Search' }).click()

    await page.getByRole('button', { name: 'Show on graph' }).click()

    await expect(page.getByRole('button', { name: 'Close detail' })).toBeVisible({
      timeout: 20_000,
    })
    await expect(page.getByText('insight:b72dafa3f874480d', { exact: false })).toBeVisible()
  })
})
