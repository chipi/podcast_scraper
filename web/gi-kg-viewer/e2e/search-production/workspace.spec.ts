import { expect, test } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  resetUserPreferences,
  SHELL_HEADING_RE,
  signInIsolated,
  statusBarCorpusPathInput,
} from '../helpers'

/**
 * Search v3 Tier-2 — Query Workspace end-to-end walk (RFC-107, ADR-095 §Tier-2).
 *
 * This spec exercises the entire Search v3 arc in one flow:
 *
 *   S2 (workspace) → S1 (chip filters) → S4b (operator=cluster server)
 *      → S5 (enriched hero from decorated hits) → S4a (Timeline)
 *      → S7 (Recent auto-write) → S3 (palette rehydration)
 *
 * The point of a Tier-2 walk is to catch cross-slice regressions the per-slice specs don't see —
 * Recent's ring buffer fighting the palette's live-fetch, the operator bar's over-fetch conflicting
 * with the enriched hero's aggregation, and so on.
 *
 * #1619 — migrated to the live stack, which is the whole point of a Tier-2 walk.
 *
 * The previous version was "production-shaped": a route handler that reproduced what the shipped
 * `/api/search` returns, branching on `operator` and `enrich_results`. It was carefully built and
 * still could not catch a cross-slice regression, because every slice was reading from the same
 * hand-written object — the walk exercised the mock's consistency, not the system's. It also
 * carried a standing TODO to regenerate the fixture against the shipped shape, which is exactly
 * the maintenance burden that disappears here.
 *
 * Now: one real query, one real backend, and every expectation read back from it.
 */
const QUERY = 'systems thinking'

test.describe('Search v3 Tier-2 — Query Workspace end-to-end walk', () => {
  test('Workspace walk: results → cluster → consensus → timeline → enriched hero → recent write', async ({
    page,
  }, testInfo) => {
    /* Real session with clean prefs: S7/S3 assert USERPREFS-1 writes, so a stubbed auth (no
     * server session) would 401 the store and the Recent assertions would pass vacuously. */
    await signInIsolated(page, 'tier2-workspace-walk', testInfo)
    await resetUserPreferences(page)

    // ---- S2: Land on the workspace ----
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    const workspace = page.getByTestId('search-workspace')
    await expect(workspace).toBeVisible()

    // ---- S1 + S2: Run the query ----
    await page.locator('#search-q').fill(QUERY)
    await page.locator('#search-q').press('Enter')
    await expect(workspace.locator('article').first()).toBeVisible({ timeout: 30_000 })

    // ---- S4a: Operator bar visible with the 4 chips ----
    await expect(page.getByTestId('result-set-operator-bar')).toBeVisible()
    // On-graph resolves ids from the result set; the count is ranking-dependent, so assert the
    // shape and that it resolved something.
    await expect(page.getByTestId('operator-chip-graph')).toHaveText(/On graph \(\d+\)/)

    // ---- S4b: Cluster operator fires + renders what the server grouped ----
    const clusterResp = await page.request.get(
      `/api/search?q=${encodeURIComponent(QUERY)}&operator=cluster&top_k=30`,
    )
    const { clusters } = (await clusterResp.json()) as {
      clusters: { cluster_kind: string; label: string }[]
    }
    expect(clusters.length).toBeGreaterThan(1)

    await page.getByTestId('operator-chip-cluster').click()
    const clusterPanel = page.getByTestId('operator-cluster-panel')
    await expect(clusterPanel).toBeVisible()
    const clusterRows = page.getByTestId('operator-cluster-list').locator('li')
    await expect(clusterRows).toHaveCount(clusters.length)
    await expect(clusterRows.last()).toContainText('Other') // ungrouped bucket renders last
    // Toggle off before the next operator.
    await page.getByTestId('operator-chip-cluster').click()
    await expect(clusterPanel).toHaveCount(0)

    // ---- S4b: Consensus operator fires + renders paired evidence ----
    const consensusResp = await page.request.get(
      `/api/search?q=${encodeURIComponent(QUERY)}&operator=consensus&top_k=30`,
    )
    const { consensus_pairs: pairs } = (await consensusResp.json()) as {
      consensus_pairs: { insight_a_text: string }[]
    }
    expect(pairs.length).toBeGreaterThan(0)

    await page.getByTestId('operator-chip-consensus').click()
    const consensusPanel = page.getByTestId('operator-consensus-panel')
    await expect(consensusPanel).toBeVisible()
    const consensusRows = page.getByTestId('operator-consensus-list').locator('li')
    await expect(consensusRows).toHaveCount(pairs.length)
    /* Assert the evidence text, not the speaker names: this corpus returns consensus pairs with
     * `person_a_label` / `person_b_label` null (ladder §B — v4 needs a named pair). */
    await expect(consensusRows.first()).toContainText(pairs[0]!.insight_a_text.slice(0, 40))
    // Close before moving on.
    await page.getByTestId('operator-chip-consensus').click()
    await expect(consensusPanel).toHaveCount(0)

    // ---- S4a: Timeline operator (client-only) ----
    await page.getByTestId('operator-chip-timeline').click()
    const timelinePanel = page.getByTestId('operator-timeline-panel')
    await expect(timelinePanel).toBeVisible()
    await page.getByTestId('operator-chip-timeline').click()
    await expect(timelinePanel).toHaveCount(0)

    // ---- S5: Enriched hero renders topic chips from the enrichment decoration ----
    // The initial search already fired enrich_results=true because the chip auto-adopts the
    // server capability, so hits carry query_enrichments.
    const hero = page.getByTestId('enriched-answer-hero')
    await expect(hero).toBeVisible({ timeout: 30_000 })
    const heroChips = page.getByTestId('enriched-answer-topics').locator('li')
    const chipCount = await heroChips.count()
    expect(chipCount).toBeGreaterThan(0)
    expect(chipCount).toBeLessThanOrEqual(6) // UXS-008 cap

    // ---- S7: Recent auto-populated after the search ----
    await expect(page.getByTestId('left-panel-recent-list')).toBeVisible()
    await expect(
      page.getByTestId('left-panel-recent-list').locator('button').first(),
    ).toContainText(QUERY)

    // ---- S3: Palette empty-state reads Recent from USERPREFS-1 ----
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('/')
    await expect(page.getByTestId('command-palette')).toBeVisible()
    await expect(
      page.getByTestId('command-palette-recent-list').locator('button').first(),
    ).toContainText(QUERY)
  })
})
