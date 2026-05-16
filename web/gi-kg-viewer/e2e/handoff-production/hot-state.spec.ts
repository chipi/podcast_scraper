/**
 * Tier-2 hot-state matrix against the production-shaped fixture.
 *
 * The headline row here is P2.1 — Library → Library supersession. This
 * is the V5 reproducer: the Tier-1 mock passes because layout finishes
 * in <100 ms; this tier exercises it on a 270+ node graph where the
 * supersession path is timing-sensitive.
 *
 * RFC-086 / ADR-095.
 */

import { expect, test } from '@playwright/test'
import {
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from '../helpers'
import {
  assertHandoffApplied,
  captureConsoleErrors,
  readFsmState,
} from '../handoff/_handoff-helpers'
import {
  fixtureEpisodes,
  setupProductionShapedMocks,
} from './_helpers'

test.describe('Handoff matrix § Tier 2 — Hot state (production-shaped)', () => {
  test('P2.1 — Library → Library supersession (V5 reproducer)', async ({ page }) => {
    // The smoking gun. Tier-1 H2.1 passes because the mock graph is
    // tiny; on a 270+ node real-corpus graph the second click should
    // ALSO supersede cleanly. If it stuck-timeouts (V5), this row
    // fails and forces a fix.
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible({ timeout: 15_000 })

    const eps = fixtureEpisodes()
    expect(eps.length).toBeGreaterThanOrEqual(2)

    // First click → first episode.
    const allRows = page.getByRole('button', { name: /, / })
    await allRows.first().click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()
    await assertHandoffApplied(page, `__unified_ep__:${eps[0]!.episode_id}`, {
      errors: errs,
      cameraCenterTolerance: 0.4,
      waitForReadyMs: 30_000,
    })

    // Second click → a DIFFERENT episode. Supersession should land
    // cleanly. If stuck-timeout fires, lastResult.status flips to
    // failed and the assertion catches it.
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible({ timeout: 15_000 })
    await allRows.nth(1).click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()
    await assertHandoffApplied(page, `__unified_ep__:${eps[1]!.episode_id}`, {
      errors: errs,
      cameraCenterTolerance: 0.4,
      waitForReadyMs: 30_000,
    })

    // Generation should have bumped at least twice.
    const after = await readFsmState(page)
    expect(after?.generation).toBeGreaterThanOrEqual(2)
  })

  test('P2.1-slow — Library → Library supersession with 250 ms artifact latency (V5 reproducer)', async ({ page }) => {
    // Same flow as P2.1 but with artificial 250 ms latency on every
    // artifact fetch. Simulates the real-backend timing of the
    // validation walk (V5 in real-corpus.spec.ts). If the supersession
    // path has a race that only fires when the second click happens
    // while the first's artifact load is in flight, this row catches
    // it — Tier 1 mocks finish instantly so the race never opens.
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page, { artifactLatencyMs: 250 })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible({ timeout: 15_000 })

    const eps = fixtureEpisodes()
    const allRows = page.getByRole('button', { name: /, / })
    await allRows.first().click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()
    await assertHandoffApplied(page, `__unified_ep__:${eps[0]!.episode_id}`, {
      errors: errs,
      cameraCenterTolerance: 0.4,
      waitForReadyMs: 45_000,
    })

    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible({ timeout: 15_000 })
    await allRows.nth(1).click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()
    await assertHandoffApplied(page, `__unified_ep__:${eps[1]!.episode_id}`, {
      errors: errs,
      cameraCenterTolerance: 0.4,
      waitForReadyMs: 45_000,
    })
  })

  test('P2.4 — Episode panel re-click supersession (production-shaped)', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    const eps = fixtureEpisodes()

    // Click row 1, "Open in graph", then go back, click row 2, "Open in graph".
    const allRows = page.getByRole('button', { name: /, / })
    await allRows.first().click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()
    await assertHandoffApplied(page, `__unified_ep__:${eps[0]!.episode_id}`, {
      errors: errs,
      cameraCenterTolerance: 0.4,
      waitForReadyMs: 30_000,
    })

    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await allRows.nth(1).click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()
    await assertHandoffApplied(page, `__unified_ep__:${eps[1]!.episode_id}`, {
      errors: errs,
      cameraCenterTolerance: 0.4,
      waitForReadyMs: 30_000,
    })
  })
})
