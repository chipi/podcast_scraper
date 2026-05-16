/**
 * Tier-2 cross-entry matrix (production-shaped fixture). RFC-086.
 *
 * Mirrors Tier-1 §4 (H4.1–H4.3). Composite multi-envelope sequences;
 * dev-hook driven to avoid duplicating UI walks that Tier-1 covers.
 */

import { expect, test } from '@playwright/test'
import {
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from '../helpers'
import {
  assertFsmEventEnvelope,
  captureConsoleErrors,
  readFsmEventLog,
  readFsmState,
} from '../handoff/_handoff-helpers'
import { setupProductionShapedMocks } from './_helpers'

test.describe('Handoff matrix § Tier 2 — Cross-entry (production-shaped)', () => {
  test('P4.1 — Library → Digest → Search (3 envelopes, no contamination)', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page, { search: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.waitForTimeout(800)

    const before = await readFsmState(page)
    const startGen = before?.generation ?? 0

    // Three envelopes from three different surfaces via dev-hook.
    await page.evaluate(() => {
      const store = (window as unknown as {
        __GIKG_HANDOFF_STORE__?: { handoffRequested: (e: Record<string, unknown>) => void }
      }).__GIKG_HANDOFF_STORE__
      if (!store) return
      store.handoffRequested({
        kind: 'episode',
        episodeId: '653bb844-d3b0-425e-9e28-b43e00ee5196',
        source: 'library',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
      store.handoffRequested({
        kind: 'topic',
        cyId: 'topic:public-investment',
        source: 'digest',
        loadSource: 'digest-external',
        camera: { kind: 'center-on-target' },
      })
      store.handoffRequested({
        kind: 'topic',
        cyId: 'topic:urban-planning',
        source: 'search',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
    })
    await page.waitForTimeout(1500)

    const after = await readFsmState(page)
    expect(after?.generation).toBeGreaterThanOrEqual(startGen + 3)
    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    expect(handoffs.length).toBeGreaterThanOrEqual(3)
    expect(errs.errors).toEqual([])
  })

  test('P4.3 — Search → NodeDetail → Search (graph-internal between externals)', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.waitForTimeout(800)

    await page.evaluate(() => {
      const store = (window as unknown as {
        __GIKG_HANDOFF_STORE__?: { handoffRequested: (e: Record<string, unknown>) => void }
      }).__GIKG_HANDOFF_STORE__
      if (!store) return
      store.handoffRequested({
        kind: 'topic', cyId: 'topic:public-investment', source: 'search',
        loadSource: 'subject-external', camera: { kind: 'center-on-target' },
      })
      store.handoffRequested({
        kind: 'graph-node', cyId: 'g:topic:public-investment', source: 'node-detail',
        loadSource: 'graph-internal', camera: { kind: 'center-on-target' },
      })
      store.handoffRequested({
        kind: 'topic', cyId: 'topic:urban-planning', source: 'search',
        loadSource: 'subject-external', camera: { kind: 'center-on-target' },
      })
    })
    await page.waitForTimeout(1500)

    const log = await readFsmEventLog(page)
    expect(log.filter((e) => e.type === 'handoffRequested').length).toBeGreaterThanOrEqual(3)
    await assertFsmEventEnvelope(page, {
      type: 'handoffRequested', source: 'node-detail', kind: 'graph-node',
      loadSource: 'graph-internal', cameraKind: 'center-on-target', errors: errs,
    })
  })
})
