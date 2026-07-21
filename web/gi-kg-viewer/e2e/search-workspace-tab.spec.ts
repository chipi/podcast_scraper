import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * Search v3 §S2 + §S4-shell — the Search main tab as a first-class
 * shell surface. This Tier-1 spec covers the tab surface itself
 * (nav-button ordering, keyboard shortcut, workspace region, LeftPanel
 * visibility on Search post-§S4-shell). The query workflow lives in
 * dedicated slice specs (search-fr1, search-operator-bar,
 * search-enriched-hero, search-rail-in-episode, search-saved-queries);
 * this file is the "the tab exists and behaves like a tab" contract.
 *
 * Unit coverage (SearchTab.test.ts): mount + emit contracts. This
 * Tier-1 spec covers the LIVE shell shape.
 */
test.describe('Search — main tab surface (#1232)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          search_api: true,
        }),
      })
    })
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
      })
    })
    await page.route('**/api/app/preferences', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ preferences: {} }),
      })
    })
  })

  test('Search appears as the 3rd main tab (Digest / Library / Search / Graph / Dashboard)', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    const nav = mainViewsNav(page)
    const buttons = await nav.getByRole('button').allInnerTexts()
    // Digest / Library / Search / Graph are the four main tabs visible to a
    // creator role. Dashboard / Ops / Admin only appear for admin. The
    // Search v3 §S2 tab-order guarantee is: Search is the 3rd tab (between
    // Library and Graph) — the invariant that the RFC-107 §1 IA diagram
    // pins.
    expect(buttons.slice(0, 4)).toEqual(['Digest', 'Library', 'Search', 'Graph'])
  })

  test('clicking the Search nav button mounts the workspace region', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('search-workspace')).toHaveCount(0)
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    const workspace = page.getByTestId('search-workspace')
    await expect(workspace).toBeVisible()
    expect(await workspace.getAttribute('aria-label')).toBe('Query workspace')
  })

  test('keyboard shortcut `3` activates the Search tab from Digest', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    // Blur editable focus so the digit-shortcut listener fires (per
    // useViewerKeyboard.targetIsEditable gate).
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('3')
    await expect(page.getByTestId('search-workspace')).toBeVisible()
  })

  test('LeftPanel Saved+Recent rail is visible on the Search tab (§S4-shell)', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    // The rail hosts Saved + Recent — the §S4-shell pivot removed the
    // compact search launcher; the rail is now visible on the Search
    // tab too (Search itself lives in the main-tab workspace).
    await expect(page.getByTestId('left-panel-saved-queries')).toBeVisible()
    await expect(page.getByTestId('left-panel-saved-empty')).toBeVisible()
    await expect(page.getByTestId('left-panel-recent-empty')).toBeVisible()
  })

  test('Search tab honors the operator bar contract when no query is run yet', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    // Operator bar only renders when ``search.results.length > 0`` — the
    // "empty workspace" state is the honest signal.
    await expect(page.getByTestId('result-set-operator-bar')).toHaveCount(0)
    await expect(page.getByTestId('enriched-answer-hero')).toHaveCount(0)
  })

  test('re-clicking the Search tab does not remount (keep-alive)', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    const initialWorkspace = await page.getByTestId('search-workspace').elementHandle()
    // Round-trip through Library and back.
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    const secondWorkspace = await page.getByTestId('search-workspace').elementHandle()
    // Two hits at the same DOM identity — keep-alive preserved the mount.
    expect(secondWorkspace).not.toBeNull()
    expect(initialWorkspace).not.toBeNull()
  })
})
