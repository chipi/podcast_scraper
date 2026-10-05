import { expect, test } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * UXS-001 status bar — **List** opens the corpus artifact dialog; picking artifacts and pressing
 * **Load into graph** must land on the Graph tab with those artifacts drawn. Live corpus: the
 * dialog lists what `GET /api/artifacts` really returns.
 */
test.describe('Status bar artifact list → Load into graph (UXS-001)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('List opens the dialog; Load into graph switches to Graph with the canvas drawn', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await expect(page.getByTestId('digest-root')).toBeVisible()

    await page.getByTestId('status-bar-list-artifacts').click()
    const dialog = page.getByTestId('artifact-list-dialog')
    await expect(dialog).toBeVisible()
    const load = dialog.getByRole('button', { name: 'Load into graph' })
    await dialog.getByRole('button', { name: 'None', exact: true }).click()
    await expect(load).toBeDisabled()

    // One gi artifact is enough to draw a graph and keeps the load fast.
    await dialog.getByRole('checkbox', { name: /^gi / }).first().check()
    await expect(load).toBeEnabled()
    await load.click()

    await expect(dialog).toBeHidden()
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
    await expect(page.getByTestId('digest-root')).toBeHidden()
    await expect(page.locator('.graph-canvas')).toBeVisible()
    await expect(page.getByTestId('graph-status-node-count')).toContainText(/[1-9]/)
  })

  /**
   * REAL BUG — kept asserting the correct behaviour, marked `test.fail()`.
   *
   * `StatusBar.onLoadIntoGraphFromDialog` calls `artifacts.loadSelected()`, which (unlike
   * `loadRelativeArtifacts` / `loadFromLocalFiles`) never sets `manualGraphSelection`. It then
   * emits `go-graph`; when that is the session's FIRST Graph visit, App's corpus graph sync runs,
   * sees no manual selection, and replaces the user's pick with the lens auto-load (measured: one
   * gi artifact picked, 11 episodes drawn). The next test proves the pick is honoured once the
   * Graph tab has already been opened, which isolates the first-visit sync as the cause.
   */
  test('Load into graph on the first Graph visit draws only the picked artifact', async ({
    page,
  }) => {
    test.fail()
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()

    await page.getByTestId('status-bar-list-artifacts').click()
    const dialog = page.getByTestId('artifact-list-dialog')
    await dialog.getByRole('button', { name: 'None', exact: true }).click()
    await dialog.getByRole('checkbox', { name: /^gi / }).first().check()
    await dialog.getByRole('button', { name: 'Load into graph' }).click()

    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
    // The count reads 1 for a moment before the sync replaces it — assert the SETTLED graph.
    await page.waitForLoadState('networkidle')
    await expect(page.getByTestId('graph-status-episode-count')).toHaveText('1', {
      timeout: 5_000,
    })
  })

  test('Load into graph after the Graph tab has already auto-loaded shows only the pick', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await expect(page.getByTestId('graph-status-episode-count')).toHaveText(/^([2-9]|\d{2,})$/)
    // Let the auto-load finish; an in-flight one would race the manual load below.
    await page.waitForLoadState('networkidle')
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()

    await page.getByTestId('status-bar-list-artifacts').click()
    const dialog = page.getByTestId('artifact-list-dialog')
    await dialog.getByRole('button', { name: 'None', exact: true }).click()
    await dialog.getByRole('checkbox', { name: /^gi / }).first().check()
    await dialog.getByRole('button', { name: 'Load into graph' }).click()

    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
    await page.waitForLoadState('networkidle')
    await expect(page.getByTestId('graph-status-episode-count')).toHaveText('1', {
      timeout: 5_000,
    })
  })
})
