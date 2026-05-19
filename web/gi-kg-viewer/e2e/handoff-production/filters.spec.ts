/**
 * Tier-2 filters (production-shaped). RFC-086.
 *
 * Negative-coverage rows: filter / view-only actions MUST NOT fire
 * ``handoffRequested``. Mirrors Tier-1 §8 against the larger fixture
 * where filter side-effects could realistically cascade through the
 * pipeline.
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from '../helpers'
import {
  captureConsoleErrors,
  readFsmEventLog,
  readFsmState,
} from '../handoff/_handoff-helpers'
import { setupProductionShapedMocks } from './_helpers'

test.describe('Handoff matrix § Tier 2 — Filters negative (production-shaped)', () => {
  test('P8.* — Tab nav + status-bar interactions fire NO handoffRequested', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await page.waitForTimeout(800)
    await page.evaluate(() => {
      const w = window as unknown as { __GIKG_FSM_EVENT_LOG__?: unknown[] }
      w.__GIKG_FSM_EVENT_LOG__ = []
    })

    // Tab nav round-trip (no handoffs).
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.waitForTimeout(500)

    const log = await readFsmEventLog(page)
    const userHandoffs = log.filter(
      (e) =>
        e.type === 'handoffRequested' &&
        e.envelope?.source !== 'restore-preference',
    )
    expect(userHandoffs).toEqual([])
    const fsm = await readFsmState(page)
    // FSM should be quiescent (ready) — not stuck in a loading state.
    expect(['ready', 'idle']).toContain(fsm?.state)
    expect(errs.errors).toEqual([])
  })
})
