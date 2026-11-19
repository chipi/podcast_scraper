/**
 * T1 — FSM state-walking integration test.
 *
 * The 75 FSM unit tests in `services/graphHandoffFsm.test.ts` prove transitions
 * in isolation. They do NOT prove that the FSM actually walks intermediate
 * states during a real handoff. The specific regression this test catches:
 *
 *   - **Re-introduction of the `recordApplied` shortcut.** F2 removed an
 *     `if (graphHandoff.pending) recordApplied(...)` call from
 *     `finishLayoutPass` that previously fast-pathed `loading_fetch → ready`
 *     in one step. If a future PR re-adds it (or any equivalent shortcut),
 *     the FSM would bypass the pipeline entirely. This test fails by
 *     observing `[loading_fetch, ready]` with no intermediate state.
 *
 * The test is robust to the case where the mock pipeline doesn't fully
 * complete (then the FSM stays in `loading_*` or transitions via the 5s
 * stuck timer; both are acceptable). It only fails on the specific shortcut
 * regression.
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from '../helpers'
import {
  readFsmStateHistory,
  resetFsmStateHistory,
  setupHandoffMatrixMocks,
} from './_handoff-helpers'

const INTERMEDIATE_STATES = [
  'loading_bootstrap',
  'loading_merge',
  'redrawing_incremental',
  'redrawing_full',
  'applying',
] as const

test.describe('FSM state-walking § T1', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test('Episode-panel handoff: FSM never shortcuts loading_fetch → ready', async ({
    page,
  }) => {
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: 'Mock Episode Title' }),
    ).toBeVisible()

    // Snapshot history right before the click
    await resetFsmStateHistory(page)
    await page.getByRole('button', { name: 'Open in graph' }).click()

    // Brief settle window — long enough for the FSM to make initial
    // transitions but shorter than the 5s stuck-handoff timeout (so a stuck
    // FSM stays in `loading_*` rather than racing into `ready` via the
    // stuck timer).
    await page.waitForTimeout(1500)

    const history = await readFsmStateHistory(page)
    // Sanity: handoffRequested must have transitioned out of idle/ready.
    expect(history).toContain('loading_fetch')

    // Regression detector: if the FSM reached `ready`, it must have gone
    // through at least one intermediate state. The shortcut bug would
    // produce `[loading_fetch, ready]` with no intermediate.
    const reachedReady = history[history.length - 1] === 'ready'
    if (reachedReady) {
      const hasIntermediate = history.some((s) =>
        (INTERMEDIATE_STATES as readonly string[]).includes(s),
      )
      expect(
        hasIntermediate,
        `FSM reached 'ready' without traversing any intermediate state. History: ${JSON.stringify(history)}. ` +
          'This indicates the recordApplied shortcut (or equivalent) was re-introduced; F2 removed it.',
      ).toBe(true)
    }
  })

  test('FSM exposes state history hook in dev', async ({ page }) => {
    // Smoke test that `__GIKG_FSM_STATE_HISTORY__` is populated by the
    // store's `syncReactive` calls. If a future refactor accidentally
    // removes the dev hook, T1's regression detector loses its reading
    // surface; this test goes red first.
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')

    const historyExposed = await page.evaluate(() => {
      const w = window as unknown as { __GIKG_FSM_STATE_HISTORY__?: string[] }
      return Array.isArray(w.__GIKG_FSM_STATE_HISTORY__)
    })
    expect(historyExposed).toBe(true)
  })
})
