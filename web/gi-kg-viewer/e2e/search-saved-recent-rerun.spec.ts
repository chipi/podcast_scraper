import { expect, test, type Page, type Request, type TestInfo } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  resetUserPreferences,
  SHELL_HEADING_RE,
  signInIsolated,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * UXS-016 — a Saved or Recent row in the left rail is a way back INTO Search from anywhere:
 * clicking one switches to the Search workspace, fills `#search-q` and runs the query again.
 * `search-saved-queries.spec.ts` covers the writers; this covers the readers' click-through.
 *
 * Live API + real USERPREFS-1 for the same reason as that spec: `mockSignIn` leaves the server
 * without a session, so `/api/app/preferences` would 401 and the rail would never fill.
 */

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
  await expect(page.getByTestId('search-workspace').locator('article').first()).toBeVisible({
    timeout: 30_000,
  })
}

function isSearchFor(q: string) {
  return (r: Request): boolean => {
    const u = new URL(r.url())
    return u.pathname === '/api/search' && u.searchParams.get('q') === q
  }
}

test.describe('Left rail Saved / Recent rows rerun the query (UXS-016)', () => {
  test('Recent row clicked on Digest reopens Search with the query and searches again', async ({
    page,
  }, testInfo) => {
    await signInClean(page, 'rerun-recent', testInfo)
    await openSearch(page)
    await submitFromWorkspace(page, 'systems thinking')

    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await expect(page.getByTestId('search-workspace')).toBeHidden()
    await expect(page.getByTestId('left-panel-saved-queries')).toBeVisible()

    const rerun = page.waitForRequest(isSearchFor('systems thinking'))
    await page.getByTestId('left-panel-recent-list').getByRole('button').first().click()

    await expect(page.getByTestId('search-workspace')).toBeVisible()
    await expect(page.locator('#search-q')).toHaveValue('systems thinking')
    await rerun
    await expect(page.getByTestId('search-workspace').locator('article').first()).toBeVisible({
      timeout: 30_000,
    })
  })

  test('Saved row clicked on Graph reopens Search with the query and searches', async ({
    page,
  }, testInfo) => {
    await signInClean(page, 'rerun-saved', testInfo)
    await openSearch(page)
    // Saved without ever running it: the click must be what issues the search.
    await page.locator('#search-q').fill('risk management')
    await page.getByTestId('search-save-query').click()
    await expect(page.getByTestId('search-save-query')).toContainText('Saved ✓')

    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
    await expect(page.getByTestId('left-panel-saved-queries')).toBeVisible()

    const run = page.waitForRequest(isSearchFor('risk management'))
    await page.getByTestId('left-panel-saved-list').getByRole('button').first().click()

    await expect(page.getByTestId('search-workspace')).toBeVisible()
    await expect(page.locator('#search-q')).toHaveValue('risk management')
    await run
    await expect(page.getByTestId('search-workspace').locator('article').first()).toBeVisible({
      timeout: 30_000,
    })
  })
})
