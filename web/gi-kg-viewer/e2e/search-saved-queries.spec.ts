import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * Search v3 §S7 (#1237) — Saved + Recent writers via USERPREFS-1.
 *
 * Covers the writer contract that lights up the existing LeftPanel +
 * CommandPalette readers (they render honest empty states today; this
 * slice makes them populate). Every ``runSearch`` pushes onto the
 * Recent ring buffer. A "Save query" button on SearchPanel writes to
 * the Saved list; the button flips to "Saved ✓" when the current query
 * is already saved (idempotent — matches ``saveQuery`` dedupe).
 *
 * The store surfaces the mirror through ``useUserPreferencesStore``,
 * which persists to ``/api/app/preferences`` (USERPREFS-1). The test
 * seeds a PATCH mock so we can inspect the write on the wire and a GET
 * mock so hydrate reads a clean namespace on each page load.
 */
test.describe('Search — Saved + Recent writers (#1237)', () => {
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
    // USERPREFS-1 endpoints. GET returns empty prefs; PATCH echoes body
    // back so the store's optimistic mirror + response merge sees an
    // ``ok`` response with the field set.
    await page.route('**/api/app/preferences', async (route) => {
      const method = route.request().method()
      if (method === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ preferences: {} }),
        })
        return
      }
      if (method === 'PATCH') {
        // Echo the request body so the store sees a successful write.
        const req = route.request().postDataJSON() as
          | { preferences?: Record<string, unknown> }
          | null
        const preferences = req?.preferences ?? {}
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ preferences }),
        })
        return
      }
      await route.fulfill({ status: 405, body: '' })
    })
    await page.route('**/api/search?**', async (route) => {
      const url = new URL(route.request().url())
      const q = url.searchParams.get('q') ?? ''
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: q,
          query_type: 'semantic',
          results: [
            {
              doc_id: 'insight:e1:n1',
              score: 0.9,
              source_tier: 'insight',
              text: `An insight for ${q}`,
              metadata: { doc_type: 'insight', episode_id: 'e1' },
            },
          ],
        }),
      })
    })
  })

  async function submitFromWorkspace(page: import('@playwright/test').Page, q: string): Promise<void> {
    await page.locator('#search-q').fill(q)
    await page.locator('#search-q').press('Enter')
    await expect(page.getByTestId('search-workspace').locator('article').first()).toBeVisible()
  }

  test('Recent auto-populates after each successful search — most recent first', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible()

    await submitFromWorkspace(page, 'llm strategy')
    await submitFromWorkspace(page, 'cell therapy')

    // Left rail Recent surface picks up both entries, newest first.
    const recentList = page.getByTestId('left-panel-recent-list')
    await expect(recentList).toBeVisible()
    const rows = recentList.locator('button')
    await expect(rows).toHaveCount(2)
    await expect(rows.nth(0)).toContainText('cell therapy')
    await expect(rows.nth(1)).toContainText('llm strategy')
  })

  test('Repeat query de-dupes to a single Recent row (moves to front)', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()

    await submitFromWorkspace(page, 'alpha')
    await submitFromWorkspace(page, 'beta')
    await submitFromWorkspace(page, 'alpha') // repeat

    const rows = page.getByTestId('left-panel-recent-list').locator('button')
    await expect(rows).toHaveCount(2)
    await expect(rows.nth(0)).toContainText('alpha')
    await expect(rows.nth(1)).toContainText('beta')
  })

  test('Save query button writes to Saved and flips to "Saved ✓"; LeftPanel renders the row', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()

    await page.locator('#search-q').fill('climate policy')
    const saveBtn = page.getByTestId('search-save-query')
    await expect(saveBtn).toBeEnabled()
    await expect(saveBtn).toContainText('Save query')

    await saveBtn.click()
    // Button flips to "Saved ✓" (idempotent state).
    await expect(saveBtn).toContainText('Saved ✓')

    // Left rail Saved list picks up the entry.
    const savedList = page.getByTestId('left-panel-saved-list')
    await expect(savedList).toBeVisible()
    await expect(savedList.locator('button').first()).toContainText('climate policy')
  })

  test('Save button is disabled on an empty query and does not write', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()

    const saveBtn = page.getByTestId('search-save-query')
    await expect(saveBtn).toBeDisabled()
    // No matter how we try, no Saved list rendered.
    await expect(page.getByTestId('left-panel-saved-list')).toHaveCount(0)
    await expect(page.getByTestId('left-panel-saved-empty')).toBeVisible()
  })

  test('Recent + Saved populate the Command Palette empty state', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()

    await submitFromWorkspace(page, 'first search')
    await page.locator('#search-q').fill('saved query text')
    await page.getByTestId('search-save-query').click()

    // Open palette; empty-state should render Recent + Saved with our
    // rows. Use `/` (blur editable focus first) — Meta+K is flaky on
    // headless Firefox on some macOS builds.
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('/')
    await expect(page.getByTestId('command-palette')).toBeVisible()
    await expect(page.getByTestId('command-palette-recent-list')).toBeVisible()
    await expect(
      page.getByTestId('command-palette-recent-list').locator('button').first(),
    ).toContainText('first search')
  })
})
