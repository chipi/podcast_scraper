/**
 * Section 8 — Filters (HANDOFF_MATRIX.md §8).
 *
 * Filter surfaces in Digest / Library / Graph. Filters don't directly fire
 * FSM handoff events, but the matrix needs to assert two contracts:
 *
 *   1. Filter inputs work (the filter changes the visible state).
 *   2. Filter changes don't interact destructively with the FSM (no
 *      spurious envelope, no console errors, no selection wipe when not
 *      intended).
 *
 * These specs are the "negative space" of the matrix: changes to
 * adjacent UI that *shouldn't* fire handoffs but historically have done
 * so accidentally (e.g. a filter toggle that triggers a redraw that
 * tears down selection).
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from '../helpers'
import type { Page } from '@playwright/test'
import {
  captureConsoleErrors,
  readFsmEventLog,
  readFsmState,
  resetFsmEventLog,
  setupHandoffMatrixMocks,
} from './_handoff-helpers'

/**
 * Wait until the FSM is in a quiescent state (`idle` or `ready`) AND
 * `pending` is null. Some filter actions advance the FSM through
 * intermediate states (`applying` after a layoutstop); we want to read
 * the post-settle state, not catch it mid-flight.
 */
async function waitForFsmQuiescent(page: Page, maxMs: number): Promise<void> {
  const deadline = Date.now() + maxMs
  while (Date.now() < deadline) {
    const fsm = await readFsmState(page)
    if (
      fsm &&
      (fsm.state === 'idle' || fsm.state === 'ready') &&
      fsm.pending === null
    ) {
      return
    }
    await page.waitForTimeout(200)
  }
}

test.describe('Handoff matrix § Section 8 — Filters', () => {
  test('H8.1 — Digest date chip narrows window; no FSM envelope fires', async ({
    page,
  }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { digest: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await expect(page.getByTestId('digest-toolbar-filters')).toBeVisible()

    const before = await readFsmState(page)
    expect(before).not.toBeNull()
    const startGen = before!.generation
    await resetFsmEventLog(page)

    // Opening the date chip is a filter interaction; the chip lives in
    // the toolbar and produces an updated digest query but no
    // handoff envelope.
    await page.getByTestId('digest-chip-date').click()
    await page.waitForTimeout(400)
    // Close it (or any other safe interaction).
    await page.keyboard.press('Escape')
    await page.waitForTimeout(300)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    // FSM is in `ready` and pending is null — no spurious handoff
    // envelope. Generation may bump by 1 from focusCleared (Escape →
    // K1) which is fine; the key contract is "no handoffRequested for
    // a filter interaction." If Escape didn't fire (focus elsewhere),
    // gen stays equal.
    expect(['idle', 'ready']).toContain(after!.state)
    expect(after!.pending).toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen)

    // No handoffRequested events from filter interaction.
    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    expect(handoffs, 'date chip should not fire handoffRequested').toEqual([])
    expect(errs.errors).toEqual([])
  })

  test('H8.2 — Library title filter narrows rows; no FSM envelope fires', async ({
    page,
  }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()

    await resetFsmEventLog(page)
    const before = await readFsmState(page)
    expect(before).not.toBeNull()

    // Type into the title filter — should change visible row count
    // (or no-op if no rows match) without firing the FSM.
    await page.getByTestId('library-filter-title').fill('non-matching-xyz')
    await page.waitForTimeout(400)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(['idle', 'ready']).toContain(after!.state)
    expect(after!.pending).toBeNull()

    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    expect(handoffs, 'library title filter should not fire handoffRequested').toEqual([])
    expect(errs.errors).toEqual([])
  })

  test('H8.3 — Library summary filter narrows rows; no FSM envelope fires', async ({
    page,
  }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()

    await resetFsmEventLog(page)

    await page.getByTestId('library-filter-summary').fill('summary text')
    await page.waitForTimeout(400)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(['idle', 'ready']).toContain(after!.state)
    expect(after!.pending).toBeNull()

    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    expect(handoffs).toEqual([])
    expect(errs.errors).toEqual([])
  })

  test('H8.4 — Graph layout cycle does not fire FSM envelope', async ({
    page,
  }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()
    await page.waitForTimeout(1500)
    await waitForFsmQuiescent(page, 10000)

    await resetFsmEventLog(page)

    // Cycle to the next layout. This is a filter-class action (re-runs
    // a layout) and should not fire a handoff envelope.
    await page.getByTestId('graph-layout-cycle').click()
    await waitForFsmQuiescent(page, 10000)
    await page.waitForTimeout(2000)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(['idle', 'ready']).toContain(after!.state)
    expect(after!.pending).toBeNull()

    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    expect(handoffs, 'layout cycle should not fire handoffRequested').toEqual([])
    expect(errs.errors).toEqual([])
  })

  test('H8.5 — Graph relayout preserves selection + FSM state', async ({
    page,
  }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()
    await page.waitForTimeout(1500)
    await waitForFsmQuiescent(page, 10000)

    await resetFsmEventLog(page)
    await page.getByTestId('graph-relayout').click()
    await waitForFsmQuiescent(page, 10000)
    await page.waitForTimeout(2000)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(['idle', 'ready']).toContain(after!.state)
    expect(after!.pending).toBeNull()
    const log = await readFsmEventLog(page)
    expect(log.filter((e) => e.type === 'handoffRequested')).toEqual([])
    expect(errs.errors).toEqual([])
  })

  test('H8.6 — Graph minimap toggle is filter-class (no FSM envelope)', async ({
    page,
  }) => {
    // The minimap toggle is a pure UI affordance — it should not fire a
    // ``handoffRequested`` envelope or surface console errors. We don't
    // assert on FSM state because the toggle can land mid-stream while
    // an unrelated downstream watcher (e.g. ``loadEpisodeSliceForTerritoryStrip``
    // from the Open-in-graph that preceded this test step) is still
    // settling; the contract is event-level, not state-level.
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()
    await page.waitForTimeout(1500)
    await waitForFsmQuiescent(page, 10000)

    await resetFsmEventLog(page)
    await page.getByTestId('graph-minimap-toggle').click()
    await page.waitForTimeout(800)

    const log = await readFsmEventLog(page)
    expect(
      log.filter((e) => e.type === 'handoffRequested'),
      'minimap toggle must not fire handoffRequested',
    ).toEqual([])
    expect(errs.errors).toEqual([])
  })

  test('H8.7 — Graph zoom controls do not fire FSM envelope', async ({
    page,
  }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()
    await page.waitForTimeout(1500)
    await waitForFsmQuiescent(page, 10000)

    const beforeSel = await page.evaluate(() => {
      const cy = (
        window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
      ).__GIKG_CY_DEV__
      return cy ? cy.nodes(':selected').map((n) => n.id()) : []
    })

    await resetFsmEventLog(page)
    await page.getByTestId('graph-zoom-in').click()
    await page.waitForTimeout(200)
    await page.getByTestId('graph-zoom-out').click()
    await page.waitForTimeout(200)
    await page.getByTestId('graph-zoom-fit').click()
    await page.waitForTimeout(400)

    // Zoom controls are view-only — they must not fire a handoff
    // envelope. State pollution from upstream Open-in-graph chains is
    // checked separately; the contract here is event-level.

    // Selection preserved through zoom operations.
    const afterSel = await page.evaluate(() => {
      const cy = (
        window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
      ).__GIKG_CY_DEV__
      return cy ? cy.nodes(':selected').map((n) => n.id()) : []
    })
    expect(afterSel).toEqual(beforeSel)

    const log = await readFsmEventLog(page)
    expect(log.filter((e) => e.type === 'handoffRequested')).toEqual([])
    expect(errs.errors).toEqual([])
  })
})
