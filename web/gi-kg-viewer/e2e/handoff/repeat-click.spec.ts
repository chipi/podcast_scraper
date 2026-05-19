/**
 * Section 3 — Repeated click on same target (HANDOFF_MATRIX.md §3).
 *
 * Tests idempotence and the "queue same-target" re-entrance policy
 * (FSM design / decision #5).
 */

import { expect, test } from '@playwright/test'
import { loadGraphViaFilePicker, mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from '../helpers'
import {
  assertHandoffApplied,
  captureConsoleErrors,
  readFsmState,
  setupHandoffMatrixMocks,
} from './_handoff-helpers'

test.describe('Handoff matrix § Section 3 — Repeated click', () => {
  test('H3.1 — Library row × 2 (same episode) [F4d]', async ({ page }) => {
    // F4d — handoffRequested always supersedes; second click bumps generation.
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

    // Click Open in graph twice in succession.
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

  test('H3.2 — Digest pill × 2 (same topic) [F4d]', async ({ page }) => {
    // Two rapid clicks on the same digest pill bump generation by 2 each
    // time — FSM "always supersede" re-entrance policy for handoffRequested.
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
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await page
      .getByRole('button', { name: 'Open graph for topic: CI Policy' })
      .click()

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(startGen + 2)
    await assertHandoffApplied(page, 'g:topic:ci-policy', { errors: errs })
  })

  test('H3.3 — Canvas tap fires canvasTapped on FSM [F1.2]', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await loadGraphViaFilePicker(page)

    const before = await readFsmState(page)
    expect(before).not.toBeNull()

    const tapped = await page.evaluate(() => {
      const cy = (
        window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
      ).__GIKG_CY_DEV__
      if (!cy) return false
      const first = cy.nodes().first()
      if (first.empty()) return false
      first.trigger('onetap')
      return true
    })
    expect(tapped).toBe(true)
    await page.evaluate(() => {
      const cy = (
        window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
      ).__GIKG_CY_DEV__
      cy?.nodes().first().trigger('onetap')
    })

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThan(before!.generation)
    expect(errs.errors).toEqual([])
  })

  test('H3.4 — Double-tap expand fires expansionRequested [F1.3]', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await loadGraphViaFilePicker(page)

    const before = await readFsmState(page)
    expect(before).not.toBeNull()

    const tapped = await page.evaluate(() => {
      const cy = (
        window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
      ).__GIKG_CY_DEV__
      if (!cy) return false
      const first = cy.nodes().first()
      if (first.empty()) return false
      first.trigger('dbltap')
      return true
    })
    expect(tapped).toBe(true)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThan(before!.generation)
    expect(errs.errors).toEqual([])
  })
})
