/**
 * Section 4 — Cross-entry sequences (HANDOFF_MATRIX.md §4).
 *
 * Realistic user flows touching multiple entry points in sequence. Tests "no
 * state-contamination between entry points" — matches Pre-Fix Scenario 8 in
 * INCREMENTAL_LOADING_TEST_CRITERIA.md.
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from '../helpers'
import {
  captureConsoleErrors,
  readFsmState,
  setupHandoffMatrixMocks,
} from './_handoff-helpers'

test.describe('Handoff matrix § Section 4 — Cross-entry sequences', () => {
  test('H4.1 — Library → Digest → Search [F4d]', async ({ page }) => {
    // Three-surface sequence: Library row → Digest pill → Search "Show on graph".
    // Each handoff bumps generation by ≥1. Load-source tracking per
    // envelope is verified in the contracts.spec.ts (T3) tests; here we
    // pin the no-contamination contract: three clicks ⇒ generation goes
    // up by ≥3, no console errors leak across surfaces.
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { digest: true, search: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()

    const before = await readFsmState(page)
    expect(before).not.toBeNull()
    const startGen = before!.generation

    // 1. Library
    await page.getByRole('button', { name: 'Open in graph' }).click()
    await page.waitForTimeout(700)

    // 2. Digest
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await page
      .getByRole('button', { name: 'Open graph for topic: CI Policy' })
      .click()
    await page.waitForTimeout(700)

    // 3. Search (drive via dev hook — same FSM contract as H1.5; building
    //    out a search-UI fixture in addition to digest is redundant for
    //    the no-contamination invariant we're testing here).
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
        cyId: 'topic:ci-policy',
        source: 'search',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
    })
    await page.waitForTimeout(300)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 3)
    expect(errs.errors).toEqual([])
  })

  test('H4.2 — Digest band → Library row → Digest pill [F4d]', async ({ page }) => {
    // Three-surface sequence with two-from-digest. Verifies CameraStrategy
    // transitions (band fit → library center-on-target → pill center) don't
    // get tangled.
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { digest: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()

    const before = await readFsmState(page)
    expect(before).not.toBeNull()
    const startGen = before!.generation

    // 1. Digest band hit
    await page
      .getByRole('button', { name: /Open graph and episode details/ })
      .first()
      .click()
    await page.waitForTimeout(700)

    // 2. Library row
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()
    await page.waitForTimeout(700)

    // 3. Digest CIL pill
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await page
      .getByRole('button', { name: 'Open graph for topic: CI Policy' })
      .click()
    await page.waitForTimeout(700)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 3)
    expect(errs.errors).toEqual([])
  })

  test('H4.3 — Search → NodeDetail Load → Search [F4d]', async ({ page }) => {
    // Three FSM events, two surfaces. Definition X: NodeDetail Load uses
    // graph-internal load-source (preserves layout); search uses
    // subject-external. Three events ⇒ generation bumps by ≥3.
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { search: true, clusters: true })
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
        cyId: 'topic:ci-policy',
        source: 'search',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
      store?.handoffRequested({
        kind: 'graph-node',
        cyId: 'tc:ci-policy-cluster',
        source: 'node-detail',
        loadSource: 'graph-internal',
        camera: { kind: 'center-on-target' },
      })
      store?.handoffRequested({
        kind: 'topic',
        cyId: 'topic:ci-policy',
        source: 'search',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
    })
    await page.waitForTimeout(400)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 3)
    expect(errs.errors).toEqual([])
  })
})
