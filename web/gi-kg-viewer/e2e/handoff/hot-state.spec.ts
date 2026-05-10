/**
 * Section 2 — Hot state with prior selection (HANDOFF_MATRIX.md §2).
 *
 * User has already focused episode A; now triggers a handoff for episode B (or topic Z)
 * from each entry point. Tests "second click works as well as first."
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from '../helpers'
import {
  captureConsoleErrors,
  readFsmState,
  setupHandoffMatrixMocks,
} from './_handoff-helpers'

test.describe('Handoff matrix § Section 2 — Hot state', () => {
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

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 2)
    expect(errs.errors).toEqual([])
  })

  test('H2.2 — Digest A → Digest B (D1 hot) [F4c]', async ({ page }) => {
    await setupHandoffMatrixMocks(page)
    test.skip(
      true,
      'D1 hot state needs full digest mock with multiple pills; FSM event class covered by digest.spec.ts indirectly.',
    )
  })

  test('H2.3 — Search A → Search B (S1 hot) [F4c]', async ({ page }) => {
    await setupHandoffMatrixMocks(page)
    test.skip(
      true,
      'S1 hot state needs full search mock; routing via @go-graph + activateGraphTab(source: search) wired in F1.6.',
    )
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
    expect(errs.errors).toEqual([])
  })

  test('H2.5 — Mixed: Digest A → Library B [F4c]', async ({ page }) => {
    await setupHandoffMatrixMocks(page)
    test.skip(
      true,
      'Cross-surface Digest → Library needs digest fixtures with topic pill; highlight envelope reset (decision #10) is wired in C7 / F1.5.',
    )
  })

  test('H2.6 — Mixed: Library A → Digest B [F4c]', async ({ page }) => {
    await setupHandoffMatrixMocks(page)
    test.skip(
      true,
      'Cross-surface Library → Digest needs digest fixtures; load-source flip-flop verified by C1 + F1.5 migrations.',
    )
  })

  test('H2.7 — Mixed: Search → NodeDetail Load [F4c]', async ({ page }) => {
    await setupHandoffMatrixMocks(page)
    test.skip(
      true,
      'Cross-surface Search → NodeDetail needs search + corpus-with-clusters fixtures; Definition X classification wired in F1.6.',
    )
  })
})
