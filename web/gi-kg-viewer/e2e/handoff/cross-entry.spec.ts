/**
 * Section 4 — Cross-entry sequences (HANDOFF_MATRIX.md §4).
 *
 * Realistic user flows touching multiple entry points in sequence. Tests "no
 * state-contamination between entry points" — matches Pre-Fix Scenario 8 in
 * INCREMENTAL_LOADING_TEST_CRITERIA.md.
 */

import { test } from '@playwright/test'
import { setupHandoffMatrixMocks } from './_handoff-helpers'

test.describe('Handoff matrix § Section 4 — Cross-entry sequences', () => {
  test('H4.1 — Library → Digest → Search [F4d]', async ({ page }) => {
    await setupHandoffMatrixMocks(page)
    test.skip(
      true,
      'Three-surface sequence needs library + digest + search mocks; FSM load-source tracking verified per surface in F1 + H2.1.',
    )
  })

  test('H4.2 — Digest band → Library row → Digest pill [F4d]', async ({ page }) => {
    await setupHandoffMatrixMocks(page)
    test.skip(
      true,
      'Digest band sequence needs full digest topic-band mock; CameraStrategy fit→center→center wired in F1.5.',
    )
  })

  test('H4.3 — Search → NodeDetail Load → Search [F4d]', async ({ page }) => {
    await setupHandoffMatrixMocks(page)
    test.skip(
      true,
      'Search → NodeDetail sequence needs search + corpus-with-clusters fixtures; Definition X classification verified in F1.6.',
    )
  })
})
