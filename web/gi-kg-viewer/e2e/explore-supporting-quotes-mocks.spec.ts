import { readFileSync } from 'node:fs'
import { expect, test } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from './fixtures'
import { mainViewsNav, SHELL_HEADING_RE } from './helpers'

const artifactJson = readFileSync(GI_SAMPLE_FIXTURE, 'utf-8')

test.describe('Explore supporting quotes (mocked API)', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          explore_api: true,
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

    await page.route('**/api/explore?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          kind: 'explore',
          data: {
            episodes_searched: 1,
            summary: {
              insight_count: 1,
              grounded_insight_count: 1,
              quote_count: 1,
              episode_count: 1,
              speaker_count: 0,
              topic_count: 0,
            },
            insights: [
              {
                insight_id: 'ins:explore-mock',
                text: 'Explore mock insight with supporting quote and no speaker fields.',
                grounded: true,
                confidence: 0.88,
                episode: { episode_id: 'ep-explore', title: 'Mock Explore Episode' },
                supporting_quotes: [
                  { text: 'Quoted line in explore response without speaker attribution.' },
                ],
              },
            ],
          },
        }),
      })
    })
  })

  test('supporting quote without speaker shows muted #541 hint', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await page.getByRole('navigation', { name: 'Right panel tabs' }).getByRole('button', { name: 'Explore' }).click()

    await page.getByRole('button', { name: 'Run explore' }).click()

    await page.getByText('Explore mock insight with supporting quote', { exact: false }).waitFor({
      timeout: 10_000,
    })

    await page.getByRole('button', { name: 'Show 1 quote' }).click()

    const hint = page.getByTestId('supporting-quote-speaker-unavailable')
    await expect(hint).toBeVisible()
    await expect(hint).toContainText('No speaker detected')
  })
})
