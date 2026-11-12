/**
 * Section 4 — Cross-entry sequences (HANDOFF_MATRIX.md §4).
 *
 * Realistic user flows touching multiple entry points in sequence. Tests "no
 * state-contamination between entry points" — matches Pre-Fix Scenario 8 in
 * INCREMENTAL_LOADING_TEST_CRITERIA.md.
 */

import { expect, test } from '@playwright/test'
import { SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from '../helpers'
import {
  assertFsmEventEnvelope,
  captureConsoleErrors,
  readFsmEventLog,
  readFsmState,
  setupHandoffMatrixMocks,
} from './_handoff-helpers'

test.describe('Handoff matrix § Section 4 — Cross-entry sequences', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test('H4.3 — Search → NodeDetail Load → Search [F4d]', async ({ page }) => {
    // Three FSM events, two surfaces. Definition X: NodeDetail Load uses
    // graph-internal load-source (preserves layout); search uses
    // subject-external. Three events ⇒ generation bumps by ≥3.
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { search: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await page.waitForTimeout(500)

    const before = await readFsmState(page)
    expect(before).not.toBeNull()
    const startGen = before!.generation

    await page.evaluate(() => {
      const store = (
        window as unknown as {
          __GIKG_HANDOFF_STORE__?: {
            handoffRequested: (env: Record<string, unknown>) => void
          }
        }
      ).__GIKG_HANDOFF_STORE__
      store?.handoffRequested({
        kind: 'topic',
        cyId: 'g:topic:ci-policy',
        source: 'search',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
      store?.handoffRequested({
        kind: 'graph-node',
        cyId: 'g:insight:b72dafa3f874480d',
        source: 'node-detail',
        loadSource: 'graph-internal',
        camera: { kind: 'center-on-target' },
      })
      store?.handoffRequested({
        kind: 'topic',
        cyId: 'g:topic:ci-policy',
        source: 'search',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
    })
    await page.waitForTimeout(400)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 3)
    // Three envelopes recorded: search → node-detail → search. Search
    // events carry ``subject-external`` (Definition X: new context);
    // node-detail carries ``graph-internal`` (preserves layout).
    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    expect(handoffs.length).toBeGreaterThanOrEqual(3)
    const searchEvents = handoffs.filter((e) => e.envelope?.source === 'search')
    expect(searchEvents.length).toBeGreaterThanOrEqual(2)
    for (const e of searchEvents) {
      expect(e.envelope?.kind).toBe('topic')
      expect(e.envelope?.loadSource).toBe('subject-external')
      expect(e.envelope?.camera?.kind).toBe('center-on-target')
    }
    await assertFsmEventEnvelope(page, {
      type: 'handoffRequested',
      source: 'node-detail',
      kind: 'graph-node',
      loadSource: 'graph-internal',
      cameraKind: 'center-on-target',
      errors: errs,
    })
  })
})
