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
 *   S2 (workspace) → S1 (chip filters) → S4a (Timeline)
 *      → S7 (Recent auto-write) → S3 (palette rehydration)
 *
 * The point of a Tier-2 walk is to catch cross-slice regressions the per-slice specs don't see —
 * Recent's ring buffer fighting the palette's live-fetch, and so on.
 *
 * #1619 — migrated to the live stack, which is the whole point of a Tier-2 walk.
 *
 * The previous version was "production-shaped": a route handler that reproduced what the shipped
 * `/api/search` returns, branching on `operator`. It was carefully built and
 * still could not catch a cross-slice regression, because every slice was reading from the same
 * hand-written object — the walk exercised the mock's consistency, not the system's. It also
 * carried a standing TODO to regenerate the fixture against the shipped shape, which is exactly
 * the maintenance burden that disappears here.
 *
 * Now: one real query, one real backend, and every expectation read back from it.
 */
const QUERY = 'systems thinking'

test.describe('Search v3 Tier-2 — Query Workspace end-to-end walk', () => {
  test('Workspace walk: results → timeline → recent write', async ({
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

    // ---- S4a: Operator bar visible ----
    await expect(page.getByTestId('result-set-operator-bar')).toBeVisible()
    // On-graph resolves ids from the result set; the count is ranking-dependent, so assert the
    // shape and that it resolved something.
    await expect(page.getByTestId('operator-chip-graph')).toHaveText(/On graph \(\d+\)/)

    // ---- S4a: Timeline operator (client-only) ----
    await page.getByTestId('operator-chip-timeline').click()
    const timelinePanel = page.getByTestId('operator-timeline-panel')
    await expect(timelinePanel).toBeVisible()
    await page.getByTestId('operator-chip-timeline').click()
    await expect(timelinePanel).toHaveCount(0)

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
