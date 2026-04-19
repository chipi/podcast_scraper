import { readFileSync } from 'node:fs'
import { expect, test } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from './fixtures'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

const artifactJson = readFileSync(GI_SAMPLE_FIXTURE, 'utf-8')

const TRANSCRIPT_BODY =
  'Hello world transcript sample for CI quality metrics fixture.\nExtra line for scroll.'

test.describe('Transcript viewer dialog (mocked API)', () => {
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
              mtime_utc: '2026-04-18T12:00:00Z',
              publish_date: '2026-04-18',
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
          body: JSON.stringify([
            { start: 0, end: 1.2, text: 'Hello world' },
            { start: 1.2, end: 3, text: ' transcript sample' },
          ]),
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

    await page.route('**/api/search?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: 'stub-quote',
          results: [
            {
              doc_id: 'doc-quote',
              score: 0.91,
              text: 'Hello world transcript sample for CI quality metrics fixture.',
              metadata: {
                doc_type: 'quote',
                source_id: 'quote:4729aa32a95c9ca1',
                episode_id: 'ci-fixture',
                source_metadata_relative_path: 'metadata/ci_sample.metadata.json',
              },
            },
          ],
        }),
      })
    })
  })

  test('Quote node detail opens transcript dialog with highlight and timeline', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await page.locator('#search-q').fill('stub quote hit')
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()

    await page.getByText('Hello world transcript sample', { exact: false }).waitFor({ timeout: 10_000 })

    await page.getByRole('button', { name: 'Show on graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    await expect(page.getByRole('region', { name: 'Graph node: Quote' })).toBeVisible({ timeout: 15_000 })

    const quoteSpeakerHint = page.getByTestId('node-detail-quote-speaker-unavailable')
    await expect(quoteSpeakerHint).toBeVisible()
    await expect(quoteSpeakerHint).toContainText('No speaker detected')

    await page.getByTestId('node-detail-view-transcript').click()

    const dlg = page.getByTestId('transcript-viewer-dialog')
    await expect(dlg).toBeVisible()
    await expect(dlg.getByRole('heading', { name: 'Transcript' })).toBeVisible()

    await expect(dlg.getByTestId('transcript-viewer-char-range')).toContainText('Characters 0')

    await expect(dlg.getByTestId('transcript-viewer-highlight')).toContainText(
      'Hello world transcript sample for CI quality metrics fixture',
    )

    await dlg.getByText('Timeline (2 segments)', { exact: false }).click()
    await expect(dlg.getByTestId('transcript-viewer-timeline')).toBeVisible()
    await expect(dlg.getByText('0.0s – 1.2s', { exact: false })).toBeVisible()
  })
})
