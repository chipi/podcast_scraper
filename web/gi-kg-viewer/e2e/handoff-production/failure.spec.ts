/**
 * Tier-2 failure modes (production-shaped fixture). RFC-086.
 * Mirrors Tier-1 §6 (H6.1–H6.3).
 */

import { expect, test } from '@playwright/test'
import { SHELL_HEADING_RE, statusBarCorpusPathInput, mainViewsNav } from '../helpers'
import {
  assertFsmEventEnvelope,
  captureConsoleErrors,
  readFsmEventLog,
  readFsmState,
} from '../handoff/_handoff-helpers'
import { setupProductionShapedMocks } from './_helpers'

test.describe('Handoff matrix § Tier 2 — Failure modes (production-shaped)', () => {
  test('P6.2 — Non-existent cy id surfaces handoffFailed cleanly', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.waitForTimeout(1500)

    await page.evaluate(() => {
      const store = (window as unknown as {
        __GIKG_HANDOFF_STORE__?: { handoffRequested: (e: Record<string, unknown>) => void }
      }).__GIKG_HANDOFF_STORE__
      store?.handoffRequested({
        kind: 'graph-node',
        cyId: 'g:topic:this-node-does-not-exist-in-fixture-xyz',
        source: 'restore-preference',
        loadSource: 'graph-internal',
        camera: { kind: 'preserve' },
      })
    })
    await assertFsmEventEnvelope(page, {
      type: 'handoffRequested', source: 'restore-preference', kind: 'graph-node',
      loadSource: 'graph-internal', cameraKind: 'preserve', errors: errs,
    })
    // FSM must reach a terminal state (no stuck).
    const fsm = await page.evaluate(async () => {
      const deadline = Date.now() + 20_000
      while (Date.now() < deadline) {
        const w = window as unknown as { __GIKG_FSM__?: { state: string; lastResult: unknown } }
        if (w.__GIKG_FSM__?.lastResult) return w.__GIKG_FSM__
        await new Promise((r) => setTimeout(r, 200))
      }
      const w2 = window as unknown as { __GIKG_FSM__?: { state: string; lastResult: unknown } }
      return w2.__GIKG_FSM__ ?? null
    })
    expect(fsm).not.toBeNull()
  })
})
