/**
 * Tier-2 repeated-click matrix (production-shaped fixture). RFC-086.
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

test.describe('Handoff matrix § Tier 2 — Repeated click (production-shaped)', () => {
  test('P3.1 — Library × 2 (same episode) supersedes', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
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
      waitForReadyMs: 30_000,
    })
    const before = await readFsmState(page)
    const startGen = before!.generation

    // Same episode, second click.
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await allRows.first().click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()
    await assertHandoffApplied(page, `__unified_ep__:${eps[0]!.episode_id}`, {
      errors: errs,
      cameraCenterTolerance: 0.4,
      waitForReadyMs: 30_000,
    })
    const after = await readFsmState(page)
    expect(after!.generation).toBeGreaterThan(startGen)
  })
})
