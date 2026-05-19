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

  test('P2.5 — Digest → Library when target artifacts already loaded (FSM apply must reach ready)', async ({ page }) => {
    // **Regression test for the L1/D1 stuck-timeout bug surfaced on real
    // corpus (Tier-3 P2.5/P2.6 batch 1).** When a prior handoff has
    // already loaded the same artifacts the new handoff would request,
    // ``appendRelativeArtifacts`` short-circuits on ``add.length === 0``
    // → no redraw fires → FSM stuck in ``loading_fetch`` → 15s timeout.
    // The fix was to follow the append with
    // ``loadSelected({preserveExpansion: true})`` so the FSM transition
    // chain runs even when no new data was added.
    //
    // Tier-2 must invoke the REAL UI click handler (LibraryView's
    // ``openEpisodeInGraph``) to exercise the fix; firing
    // ``handoffRequested`` directly via dev-hook bypasses the handler
    // and would always stuck-timeout (the fix is in the handler, not
    // the FSM transition table).
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page, { artifactLatencyMs: 50 })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')

    // Step 1: Click a Digest topic-pill to pre-load its hits' artifacts.
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    const pill = page.getByRole('button', { name: /Open graph for topic/ }).first()
    await pill.waitFor({ state: 'visible', timeout: 30_000 })
    await pill.click()
    await page.waitForTimeout(1500) // let the first handoff settle

    // Step 2: Click a Library row's G button. The episode's artifacts
    // are likely already loaded from step 1 (Digest band hits cover the
    // production-shaped fixture's episodes). Without the loadSelected
    // fix, this second click would stuck-timeout.
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.waitForTimeout(800)
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await page.waitForTimeout(3000)

    const fsm = await readFsmState(page)
    // eslint-disable-next-line no-console
    console.log('[Tier-2 P2.5]', JSON.stringify({ state: fsm?.state, lastResult: fsm?.lastResult }))
    expect(fsm).not.toBeNull()
    // FSM MUST reach ``ready`` within 3s post-click. Before the
    // ``loadSelected`` fix the FSM would still be in ``loading_fetch``
    // here (and stuck-timeout at 15s).
    expect(fsm!.state).toBe('ready')
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
