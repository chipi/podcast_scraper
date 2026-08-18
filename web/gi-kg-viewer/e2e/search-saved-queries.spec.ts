import { expect, test, type Page, type TestInfo } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  resetUserPreferences,
  SHELL_HEADING_RE,
  signInIsolated,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * Search v3 §S7 (#1237) — Saved + Recent writers via USERPREFS-1.
 *
 * Covers the writer contract that lights up the existing LeftPanel + CommandPalette readers.
 * Every ``runSearch`` pushes onto the Recent ring buffer. A "Save query" button on SearchPanel
 * writes to the Saved list; the button flips to "Saved ✓" when the current query is already saved
 * (idempotent — matches ``saveQuery`` dedupe).
 *
 * #1619 — migrated to the live API, including the persistence layer.
 *
 * The old version stubbed ``/api/app/preferences`` with a GET that returned `{}` and a PATCH that
 * echoed the body back. That is the one thing this file is actually about — whether the writer
 * round-trips through USERPREFS-1 — so echoing it back asserted nothing about persistence.
 *
 * It now runs against the real store. Two consequences shape the setup:
 *
 * * ``mockSignIn`` is not usable here: it stubs ``/api/app/auth/status`` in the browser only, so
 *   the server has no session and ``/api/app/preferences`` answers **401**. ``signInIsolated``
 *   drives the real mock-OAuth round trip. It needs the server started with
 *   ``APP_SIGNUP_MODE=open`` — without it the login endpoint 403s (see e2e/README.md).
 * * Preferences now **persist per user across tests**, so each test takes its own identity and
 *   calls ``resetUserPreferences`` — otherwise Recent from one test leaks into the next and the
 *   row-count assertions become order-dependent.
 */

/** Sign in as a per-test identity with a clean USERPREFS-1 namespace. */
async function signInClean(page: Page, who: string, testInfo: TestInfo): Promise<void> {
  await signInIsolated(page, who, testInfo)
  await resetUserPreferences(page)
}

async function openSearch(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
  await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
  await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
}

async function submitFromWorkspace(page: Page, q: string): Promise<void> {
  await page.locator('#search-q').fill(q)
  await page.locator('#search-q').press('Enter')
  await expect(
    page.getByTestId('search-workspace').locator('article').first(),
  ).toBeVisible({ timeout: 30_000 })
}

test.describe('Search — Saved + Recent writers (#1237)', () => {
  test('Recent auto-populates after each successful search — most recent first', async ({
    page,
  }, testInfo) => {
    await signInClean(page, 'saved-recent-order', testInfo)
    await openSearch(page)

    await submitFromWorkspace(page, 'systems thinking')
    await submitFromWorkspace(page, 'risk management')

    // Left rail Recent surface picks up both entries, newest first.
    const recentList = page.getByTestId('left-panel-recent-list')
    await expect(recentList).toBeVisible()
    const rows = recentList.locator('button')
    await expect(rows).toHaveCount(2)
    await expect(rows.nth(0)).toContainText('risk management')
    await expect(rows.nth(1)).toContainText('systems thinking')
  })

  test('Repeat query de-dupes to a single Recent row (moves to front)', async ({
    page,
  }, testInfo) => {
    await signInClean(page, 'saved-recent-dedupe', testInfo)
    await openSearch(page)

    await submitFromWorkspace(page, 'systems thinking')
    await submitFromWorkspace(page, 'risk management')
    await submitFromWorkspace(page, 'systems thinking') // repeat

    const rows = page.getByTestId('left-panel-recent-list').locator('button')
    await expect(rows).toHaveCount(2)
    await expect(rows.nth(0)).toContainText('systems thinking')
    await expect(rows.nth(1)).toContainText('risk management')
  })

  test('Save query button writes to Saved and flips to "Saved ✓"; LeftPanel renders the row', async ({
    page,
  }, testInfo) => {
    await signInClean(page, 'saved-write', testInfo)
    await openSearch(page)

    await page.locator('#search-q').fill('lifelong learning')
    const saveBtn = page.getByTestId('search-save-query')
    await expect(saveBtn).toBeEnabled()
    await expect(saveBtn).toContainText('Save query')

    await saveBtn.click()
    // Button flips to "Saved ✓" (idempotent state).
    await expect(saveBtn).toContainText('Saved ✓')

    // Left rail Saved list picks up the entry.
    const savedList = page.getByTestId('left-panel-saved-list')
    await expect(savedList).toBeVisible()
    await expect(savedList.locator('button').first()).toContainText('lifelong learning')

    /* The point of the migration: it really persisted. Read USERPREFS-1 back from the server
     * rather than trusting the optimistic mirror the store rendered from.
     *
     * Polled, not read once: the store writes optimistically and flushes to
     * ``/api/app/preferences`` afterwards, so a single immediate GET races the flush — the row is
     * on screen before it is on disk. */
    await expect
      .poll(
        async () => {
          const resp = await page.request.get('/api/app/preferences')
          return resp.ok() ? JSON.stringify(await resp.json()) : ''
        },
        { timeout: 15_000 },
      )
      .toContain('lifelong learning')
  })

  test('Save button is disabled on an empty query and does not write', async ({
    page,
  }, testInfo) => {
    await signInClean(page, 'saved-empty', testInfo)
    await openSearch(page)

    const saveBtn = page.getByTestId('search-save-query')
    await expect(saveBtn).toBeDisabled()
    // No matter how we try, no Saved list rendered.
    await expect(page.getByTestId('left-panel-saved-list')).toHaveCount(0)
    await expect(page.getByTestId('left-panel-saved-empty')).toBeVisible()
  })

  test('Recent + Saved populate the Command Palette empty state', async ({ page }, testInfo) => {
    await signInClean(page, 'saved-palette', testInfo)
    await openSearch(page)

    await submitFromWorkspace(page, 'expert interviews')
    await page.locator('#search-q').fill('saved query text')
    await page.getByTestId('search-save-query').click()

    // Open palette; empty-state should render Recent + Saved with our rows. Use `/` (blur
    // editable focus first) — Meta+K is flaky on headless Firefox on some macOS builds.
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('/')
    await expect(page.getByTestId('command-palette')).toBeVisible()
    await expect(page.getByTestId('command-palette-recent-list')).toBeVisible()
    await expect(
      page.getByTestId('command-palette-recent-list').locator('button').first(),
    ).toContainText('expert interviews')
  })
})
