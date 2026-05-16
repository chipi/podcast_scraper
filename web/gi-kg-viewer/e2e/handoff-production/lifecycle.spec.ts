/**
 * Tier-2 lifecycle matrix (production-shaped fixture). RFC-086.
 * Mirrors Tier-1 §7 (H7.1 restore-preference, H7.2 tab-reconcile).
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from '../helpers'
import {
  assertFsmEventEnvelope,
  captureConsoleErrors,
  readFsmState,
} from '../handoff/_handoff-helpers'
import { setupProductionShapedMocks } from './_helpers'

test.describe('Handoff matrix § Tier 2 — Lifecycle (production-shaped)', () => {
  test('P7.1 — Restore-preference envelope shape (dev-hook)', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.waitForTimeout(1000)
    await page.evaluate(() => {
      const w = window as unknown as { __GIKG_FSM_EVENT_LOG__?: unknown[] }
      w.__GIKG_FSM_EVENT_LOG__ = []
    })
    await page.evaluate(() => {
      const store = (window as unknown as {
        __GIKG_HANDOFF_STORE__?: { handoffRequested: (e: Record<string, unknown>) => void }
      }).__GIKG_HANDOFF_STORE__
      store?.handoffRequested({
        kind: 'graph-node',
        cyId: 'g:topic:public-investment',
        source: 'restore-preference',
        loadSource: 'subject-external',
        camera: { kind: 'center', cyId: 'g:topic:public-investment' },
      })
    })
    await assertFsmEventEnvelope(page, {
      type: 'handoffRequested', source: 'restore-preference', kind: 'graph-node',
      loadSource: 'subject-external', cameraKind: 'center', errors: errs,
    })
  })

  test('P7.2 — Tab-switch reconcile-only, no generation bump from tab nav', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.waitForTimeout(800)
    const before = await readFsmState(page)
    const startGen = before?.generation ?? 0

    // Tab around: Library, Digest, back to Graph.
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.waitForTimeout(500)

    const after = await readFsmState(page)
    // ±1 tolerance — App.activateGraphTab may fire once on Graph re-activation.
    expect(after?.generation).toBeLessThanOrEqual(startGen + 1)
    expect(errs.errors).toEqual([])
  })
})
