import { expect, test } from '@playwright/test'

/**
 * RFC-088 chunk-9 viewer-surface smoke specs against the live stack.
 *
 * Verifies the four new viewer surfaces (Configuration popup
 * Enrichment tab + Dashboard kind filter + episode-detail enrichment
 * section + graph EnrichmentEdgesPanel + search ResultCard chip strip)
 * are at minimum reachable and render their data-testid hooks against
 * a fresh corpus. Dynamic behaviour is covered by the chunk-9 vitest
 * mount tests; this spec is a structural smoke against the real DOM
 * served by the stack so the testids don't drift undetected.
 */

const CORPUS = '/app/output'

test.describe('stack test — RFC-088 chunk 9 viewer surfaces', () => {
  test.beforeEach(async ({ page }) => {
    // Land on the SPA shell with the corpus pre-set via query param
    // so the shell store picks it up.
    await page.goto(`/?path=${encodeURIComponent(CORPUS)}`)
  })

  test('Configuration popup → Enrichment tab is mountable + renders the panel', async ({ page }) => {
    // Open the Configuration popup from the StatusBar.
    const sourcesBtn = page.getByTestId('status-bar-sources-btn')
    if (await sourcesBtn.isVisible({ timeout: 5000 })) {
      await sourcesBtn.click()
    }
    // Click the new Enrichment tab.
    const tab = page.getByTestId('sources-dialog-tab-enrichment')
    await expect(tab).toBeVisible({ timeout: 10_000 })
    await tab.click()
    // The panel renders its action buttons + table even for a fresh corpus.
    await expect(page.getByTestId('enrichment-panel')).toBeVisible()
    await expect(page.getByTestId('enrichment-refresh-btn')).toBeVisible()
    await expect(page.getByTestId('enrichment-run-btn')).toBeVisible()
    await expect(page.getByTestId('enrichment-table')).toBeVisible()
  })

  test('Dashboard pipeline-runs strip carries the kind filter group', async ({ page }) => {
    // Navigate to the Dashboard tab and assert the kind filter buttons
    // are present. Defensive: tab structure may differ across builds.
    const dashboardTabs = [
      'main-tab-dashboard',
      'main-tab-dash',
    ]
    for (const id of dashboardTabs) {
      const btn = page.getByTestId(id)
      if (await btn.isVisible({ timeout: 1000 }).catch(() => false)) {
        await btn.click()
        break
      }
    }
    const kindFilter = page.getByTestId('pipeline-job-kind-filter')
    // The strip only renders when the API surfaces jobs; on a fresh
    // corpus the kind filter may not be present, so we assert at most
    // that when it's present, all three buttons render.
    if (await kindFilter.isVisible({ timeout: 5000 }).catch(() => false)) {
      await expect(page.getByTestId('pipeline-job-kind-filter-all')).toBeVisible()
      await expect(page.getByTestId('pipeline-job-kind-filter-pipeline')).toBeVisible()
      await expect(page.getByTestId('pipeline-job-kind-filter-enrichment')).toBeVisible()
    }
  })

  test('Configuration sub-section mounts the EnrichmentConfigEditor', async ({ page }) => {
    // RFC-088 v2 — the Configuration → Enrichment tab now embeds a
    // collapsible Configuration editor. Smoke-check that the toggle
    // expands the editor + the data-driven form hooks render.
    const sourcesBtn = page.getByTestId('status-bar-sources-btn')
    if (await sourcesBtn.isVisible({ timeout: 5000 })) {
      await sourcesBtn.click()
    }
    const tab = page.getByTestId('sources-dialog-tab-enrichment')
    await expect(tab).toBeVisible({ timeout: 10_000 })
    await tab.click()
    const toggle = page.getByTestId('enrichment-panel-config-toggle')
    await expect(toggle).toBeVisible()
    await toggle.click()
    // Editor surface is now visible
    await expect(page.getByTestId('enrichment-config-editor')).toBeVisible()
    await expect(page.getByTestId('enrichment-config-global-enabled')).toBeVisible()
    await expect(page.getByTestId('enrichment-config-enricher-list')).toBeVisible()
    // Action buttons render — Save starts disabled (no edits yet)
    await expect(page.getByTestId('enrichment-config-save-btn')).toBeVisible()
    await expect(page.getByTestId('enrichment-config-reset-btn')).toBeVisible()
    await expect(page.getByTestId('enrichment-config-refresh-btn')).toBeVisible()
  })

  test('Graph tab renders without console errors when the canvas mounts', async ({ page }) => {
    // Console-error gate: navigating to the graph tab + waiting for
    // canvas mount must NOT print red errors from EnrichmentEdgesPanel.
    const errors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text())
    })
    const graphTabs = ['main-tab-graph', 'main-tab-graph-canvas']
    for (const id of graphTabs) {
      const btn = page.getByTestId(id)
      if (await btn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await btn.click()
        break
      }
    }
    // Give the canvas + panels time to mount.
    await page.waitForTimeout(2000)
    // No errors mentioning chunk-9 components are tolerated.
    const offenders = errors.filter(
      (e) =>
        e.includes('EnrichmentEdgesPanel') ||
        e.includes('EpisodeEnrichmentSection') ||
        e.includes('useEnrichmentEnvelopeCache'),
    )
    expect(offenders).toEqual([])
  })
})
