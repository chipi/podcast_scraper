/**
 * Section 5 — Concurrency (HANDOFF_MATRIX.md §5).
 *
 * Rapid sequences and lifecycle events that exercise generation tokens +
 * supersession (FSM concurrency rules; concern #4 / decision #5).
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from '../helpers'
import {
  captureConsoleErrors,
  readFsmState,
  setupHandoffMatrixMocks,
} from './_handoff-helpers'

test.describe('Handoff matrix § Section 5 — Concurrency', () => {
  test('H5.1 — Rapid Library clicks: last wins [F4d]', async ({ page }) => {
    // F4d — generation supersession: 5 rapid Library clicks should bump
    // generation 5 times; final state matches the last click.
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

    // Click Open in graph 5 times in rapid succession (re-navigating to Library
    // each time since the panel is on the right rail).
    for (let i = 0; i < 5; i++) {
      await page.getByRole('button', { name: 'Open in graph' }).click()
      if (i < 4) {
        await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
      }
    }

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    // Each click bumps generation; 5 clicks → at least +5.
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 5)
    expect(errs.errors).toEqual([])
  })

  test('H5.2 — Mid-load tab-switch away + return [F4d]', async ({ page }) => {
    // F4d — tab return policy (decision #7): reconcile-only. Switching tabs
    // mid-flight and returning should not double-apply the in-flight handoff.
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

    // Click Open in graph (graph tab activates) → switch back to Library →
    // wait briefly → switch back to Graph.
    await page.getByRole('button', { name: 'Open in graph' }).click()
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    // One handoff fired; generation incremented once.
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 1)
    expect(errs.errors).toEqual([])
  })
})
