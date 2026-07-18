/**
 * Section 2 — Hot state with prior selection (HANDOFF_MATRIX.md §2).
 *
 * User has already focused episode A; now triggers a handoff for episode B (or topic Z)
 * from each entry point. Tests "second click works as well as first."
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from '../helpers'
import {
  assertFsmEventEnvelope,
  assertHandoffApplied,
  captureConsoleErrors,
  readFsmEventLog,
  readFsmState,
  setupHandoffMatrixMocks,
} from './_handoff-helpers'

test.describe('Handoff matrix § Section 2 — Hot state', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test('H2.1 — Library row re-click: generation supersedes [F4c]', async ({ page }) => {
    // F4c — H2.1: the canonical "second Library G" reproducer. With C1's
    // synchronous setLoadSource patch + F1.1 FSM event, repeat clicks bump
    // generation each time.
    const errs = captureConsoleErrors(page)
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

    const before = await readFsmState(page)
    expect(before).not.toBeNull()
    const startGen = before!.generation

    // First "Open in graph"
    await page.getByRole('button', { name: 'Open in graph' }).click()
    // Switch back to Library tab and re-click Open in graph
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()

    // After two clicks: generation up by ≥2 (supersession). Final state
    // applied to the same episode (same target each time).
    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 2)
    await assertHandoffApplied(page, 'g:episode:ci-fixture', {
      errors: errs,
      episodePanelTitle: 'Mock Episode Title',
    })
  })

  test('H2.2 — Digest A → Digest B (D1 hot) [F4c]', async ({ page }) => {
    // Hot-state D1 re-click: each click on the same pill bumps generation
    // (FSM "always supersede" re-entrance policy for handoffRequested).
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { digest: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()

    const before = await readFsmState(page)
    expect(before).not.toBeNull()
    const startGen = before!.generation

    await page
      .getByRole('button', { name: 'Open graph for topic: CI Policy' })
      .click()
    await page.waitForTimeout(600)
    // Switch back to Digest and re-click the same pill.
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await page
      .getByRole('button', { name: 'Open graph for topic: CI Policy' })
      .click()

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 2)
    await assertHandoffApplied(page, 'g:topic:ci-policy', { errors: errs })
  })

  test('H2.3 — Search A → Search B (S1 hot) [F4c]', async ({ page }) => {
    // Hot-state S1 re-click: two ``handoffRequested(source: 'search')`` events
    // bump generation by 2. Drive via the FSM dev hook (same contract as
    // H1.5; no search-UI fixture needed for the FSM-side assertion).
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
      const env = (cyId: string): Record<string, unknown> => ({
        kind: 'topic',
        cyId,
        source: 'search',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
      store?.handoffRequested(env('topic:ci-policy'))
      store?.handoffRequested(env('topic:ci-policy'))
    })
    await page.waitForTimeout(300)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 2)
    // 2 search envelopes fired with the same source/kind/loadSource.
    const log = await readFsmEventLog(page)
    const searchEvents = log.filter(
      (e) => e.type === 'handoffRequested' && e.envelope?.source === 'search',
    )
    expect(searchEvents.length).toBeGreaterThanOrEqual(2)
    for (const e of searchEvents) {
      expect(e.envelope?.kind).toBe('topic')
      expect(e.envelope?.loadSource).toBe('subject-external')
      expect(e.envelope?.camera?.kind).toBe('center-on-target')
    }
    expect(errs.errors).toEqual([])
  })

  test('H2.4 — Episode panel re-click: generation supersedes [F1.1]', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: 'Mock Episode Title' }),
    ).toBeVisible()

    const before = await readFsmState(page)
    expect(before).not.toBeNull()
    const startGen = before!.generation

    await page.getByRole('button', { name: 'Open in graph' }).click()
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 2)
    await assertHandoffApplied(page, 'g:episode:ci-fixture', {
      errors: errs,
      episodePanelTitle: 'Mock Episode Title',
    })
  })

  test('H2.5 — Mixed: Digest A → Library B [F4c]', async ({ page }) => {
    // Cross-surface: digest pill then library "Open in graph".
    // Each click should bump generation. Highlights from the digest envelope
    // should NOT bleed into the library envelope (decision #10: apply phase
    // resets highlights from envelope).
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { digest: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()

    const before = await readFsmState(page)
    expect(before).not.toBeNull()
    const startGen = before!.generation

    await page
      .getByRole('button', { name: 'Open graph for topic: CI Policy' })
      .click()
    await page.waitForTimeout(800)

    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 2)
    // Final action was Library Open-in-graph → episode is the target.
    await assertHandoffApplied(page, 'g:episode:ci-fixture', {
      errors: errs,
      episodePanelTitle: 'Mock Episode Title',
    })
  })

  test('H2.6 — Mixed: Library A → Digest B [F4c]', async ({ page }) => {
    // Cross-surface reverse: library Open in graph then digest pill.
    // Each click should bump generation; load-source flip-flop (subject-external
    // → digest-external) should not leak between FSM events.
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { digest: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: 'Mock Episode Title' }),
    ).toBeVisible()

    const before = await readFsmState(page)
    expect(before).not.toBeNull()
    const startGen = before!.generation

    await page.getByRole('button', { name: 'Open in graph' }).click()
    await page.waitForTimeout(800)

    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await page
      .getByRole('button', { name: 'Open graph for topic: CI Policy' })
      .click()

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 2)
    // Final action was Digest CIL pill → topic is the target.
    await assertHandoffApplied(page, 'g:topic:ci-policy', { errors: errs })
  })

  test('H2.7 — Mixed: Search → NodeDetail Load [F4c]', async ({ page }) => {
    // Cross-surface: search handoff (subject-external) followed by NodeDetail
    // Load handoff (graph-internal). Definition X: graph-internal load
    // sources preserve layout. Drive both through the dev hook to pin the
    // FSM-side contract.
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
    })
    await page.waitForTimeout(300)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 2)
    // Both envelopes recorded with the right shape: search uses
    // ``subject-external``; node-detail uses ``graph-internal`` (Definition X:
    // preserves layout). Camera strategy carries through unchanged.
    await assertFsmEventEnvelope(page, {
      type: 'handoffRequested',
      source: 'search',
      kind: 'topic',
      loadSource: 'subject-external',
      cameraKind: 'center-on-target',
      errors: errs,
    })
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
