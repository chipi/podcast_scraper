import { expect, test, type Page } from '@playwright/test'
import { mockSignIn, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

/**
 * RFC-118 — enrichment freshness widget + the FULL re-derive lever in the
 * Configuration Enrichment section.
 *
 * #1619 category C — permanently mocked, and correctly so: the widget's states
 * (recommended-with-reasons vs current) are transient corpus conditions a live
 * fixture cannot be asked for on demand, and the force lever live would run a
 * full corpus enrichment pass mid-suite.
 */

const STATS_RECOMMENDED = {
  reenrich_recommended: true,
  reenrich_reasons: ['last_run_failed_or_timed_out', 'corpus_artifacts_newer'],
  enrichers: [
    {
      enricher_id: 'topic_consensus',
      scope: 'corpus',
      stale: true,
      reasons: ['last_run_failed_or_timed_out'],
      last_status: 'timeout',
      last_computed_at: '2026-08-23T18:50:00Z',
      current_version: '2.0.0',
      output_version: '2.0.0',
    },
    {
      enricher_id: 'grounding_rate',
      scope: 'corpus',
      stale: false,
      reasons: [] as string[],
      last_status: 'ok',
      last_computed_at: '2026-08-23T18:50:00Z',
      current_version: '1.0.0',
      output_version: '1.0.0',
    },
  ],
  artifact_newest_mtime: '2026-08-23T19:00:00Z',
  last_run_status: 'failed',
  last_run_finished_at: '2026-08-23T18:50:00Z',
  corpus_path: '/mock/corpus',
}

const STATS_CURRENT = {
  ...STATS_RECOMMENDED,
  reenrich_recommended: false,
  reenrich_reasons: [] as string[],
  enrichers: [STATS_RECOMMENDED.enrichers[1]],
  last_run_status: 'ok',
}

async function setupEnrichmentRoutes(page: Page, stats: unknown): Promise<void> {
  await page.route('**/api/health**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      // feeds_api/operator_config_api gate the status-bar Configuration trigger.
      body: JSON.stringify({
        status: 'ok',
        corpus_library_api: true,
        corpus_digest_api: true,
        feeds_api: true,
        operator_config_api: true,
      }),
    })
  })
  await page.route('**/api/enrichment/stats**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(stats),
    })
  })
  await page.route('**/api/enrichment/status**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ available: false, reason: 'no status yet' }),
    })
  })
  await page.route('**/api/enrichment/health**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ enrichers: {} }),
    })
  })
  await page.route('**/api/enrichment/metrics**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ window: '24h', per_enricher: {} }),
    })
  })
  await page.route('**/api/enrichment/run-summary**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ available: false }),
    })
  })
  await page.route('**/api/corpus/enrichments**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ enrichments: [] }),
    })
  })
}

async function openEnrichmentSection(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill('/mock/corpus')
  // Enter commits the path (hasCorpusPath gates the Configuration trigger).
  await statusBarCorpusPathInput(page).press('Enter')
  await page.getByTestId('status-bar-sources-trigger').click()
  await page.getByTestId('sources-dialog-tab-enrichment').click()
  await expect(page.getByTestId('enrichment-panel')).toBeVisible()
}

test.describe('Enrichment freshness widget (RFC-118, mocked API)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin')
  })

  test('recommended state: reasons shown, Full re-enrich POSTs force=true', async ({ page }) => {
    await setupEnrichmentRoutes(page, STATS_RECOMMENDED)
    await page.route('**/api/jobs/enrichment**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.continue()
        return
      }
      await route.fulfill({
        status: 202,
        contentType: 'application/json',
        body: JSON.stringify({ job_id: 'j-rfc118', status: 'queued', corpus_path: '/mock/corpus' }),
      })
    })

    await openEnrichmentSection(page)

    const block = page.getByTestId('enrichment-freshness-block')
    await expect(block).toBeVisible()
    await expect(block).toContainText('Re-enrich recommended')
    await expect(block).toContainText('last_run_failed_or_timed_out')
    // Per-enricher rows: the stale one and the fresh one, with their statuses.
    await expect(page.getByTestId('enrichment-freshness-row-topic_consensus')).toContainText(
      'timeout',
    )
    await expect(page.getByTestId('enrichment-freshness-row-grounding_rate')).toContainText('ok')

    const reqPromise = page.waitForRequest(
      (req) => req.url().includes('/api/jobs/enrichment') && req.method() === 'POST',
    )
    await page.getByTestId('enrichment-full-reenrich-btn').click()
    const req = await reqPromise
    expect(req.postDataJSON()).toMatchObject({ force: true })
    // The submit notice surfaces the accepted job.
    await expect(page.getByTestId('enrichment-submit-notice')).toContainText('j-rfc118')
  })

  test('current state: quiet line, no force lever', async ({ page }) => {
    await setupEnrichmentRoutes(page, STATS_CURRENT)
    await openEnrichmentSection(page)

    const block = page.getByTestId('enrichment-freshness-block')
    await expect(block).toBeVisible()
    await expect(block).toContainText('Enrichment current')
    await expect(page.getByTestId('enrichment-full-reenrich-btn')).toHaveCount(0)
  })
})
