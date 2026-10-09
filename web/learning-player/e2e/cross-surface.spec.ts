import { expect, test, type Page } from '@playwright/test'
import { listenToOne, navTo, signInIsolated } from './helpers'

/**
 * A write on one surface, read on ANOTHER that was already open (2026-10-09).
 *
 * ## Why this file exists
 *
 * Ten bugs fixed on 2026-10-09 had one shape: a change made on one surface, and a different surface —
 * kept alive in the background, or reading a store nobody told — went on showing what it had before.
 * None was caught, because the suite navigates with `page.goto`: a full load rebuilds every store and
 * remounts every tab, so a stale copy can never be seen. `collections.spec.ts` ran bug #1's exact flow
 * and passed for exactly that reason. (docs/wip/e2e-cross-surface-gaps-2026-10-09.md)
 *
 * ## The rule here
 *
 * Every test has the same three steps and NO `page.goto` after sign-in:
 *   1. open the READER surface first, so it is mounted and kept alive,
 *   2. go elsewhere in-app and make the change the way a person does,
 *   3. come back in-app and assert the change shows.
 * `e2e/CONSISTENCY_MATRIX.md` lists which write each test covers.
 */

/** Library's Boards tab, reached in-app. */
async function openBoards(page: Page): Promise<void> {
  await navTo(page, 'library')
  await page.getByTestId('library-tab-collections').click()
}

/** Create a board from Library's own form. */
async function createBoardInLibrary(page: Page, name: string): Promise<void> {
  await openBoards(page)
  const form = page.locator('form').filter({ has: page.getByPlaceholder(/board name/i) })
  await form.getByPlaceholder(/board name/i).fill(name)
  await form.locator('button[type="submit"]').click()
  await expect(page.getByTestId('collection-open').filter({ hasText: name })).toBeVisible()
}

test.describe('cross-surface: a change shows on the surface that was already open', () => {
  test('a board made from the Save sheet shows in Library, with its item (#1)', async ({ page }, testInfo) => {
    await signInIsolated(page, 'xs-board-sheet', testInfo)
    const name = `Sheet ${Date.now().toString(36)}`

    await openBoards(page) // 1. the reader, mounted and kept alive

    await navTo(page, 'catalog') // 2. the write, elsewhere
    const row = page.locator('article').first()
    await expect(row).toBeVisible()
    await row.getByTestId('overflow-trigger').click()
    await page.getByTestId('add-to-collection').click()
    const menu = page.getByTestId('add-to-collection-menu')
    await menu.getByTestId('add-to-collection-new').click()
    await menu.locator('input').fill(name)
    await menu.locator('form button[type="submit"]').click()
    await expect(menu).toBeHidden({ timeout: 5000 })
    await page.keyboard.press('Escape')

    await openBoards(page) // 3. back, in-app
    const board = page.getByTestId('collection-open').filter({ hasText: name })
    await expect(board, 'the board made in the sheet is missing from Library').toBeVisible()
    await expect(board).toContainText('1 item')
  })

  test("a board deleted in Library leaves Home's teaser (#7)", async ({ page }, testInfo) => {
    await signInIsolated(page, 'xs-board-teaser', testInfo)
    const name = `Teaser ${Date.now().toString(36)}`

    await createBoardInLibrary(page, name)
    await navTo(page, 'home') // the reader: Home's boards teaser
    const tile = page.getByTestId('home-collection-tile').filter({ hasText: name })
    await expect(tile, 'a board made in Library is missing from Home').toBeVisible()

    await openBoards(page)
    await page
      .locator('[data-board-row]')
      .filter({ hasText: name })
      .getByTestId('collection-delete')
      .click()
    await page.getByTestId('confirm-accept').click()
    await expect(page.getByTestId('collection-open').filter({ hasText: name })).toHaveCount(0)

    await navTo(page, 'home')
    await expect(tile, 'a deleted board is still on Home').toHaveCount(0)
  })

  test('Your Week appears on a kept-alive Home after a listen (#4)', async ({ page }, testInfo) => {
    await signInIsolated(page, 'xs-your-week', testInfo)
    await navTo(page, 'home') // the reader; a fresh account has no week yet
    await expect(page.getByTestId('your-week')).toHaveCount(0)

    await navTo(page, 'library') // away…
    await listenToOne(page) // …a listen lands (as from another device)

    await navTo(page, 'home')
    await expect(page.getByTestId('your-week'), 'Home kept the week from before the listen').toBeVisible()
  })

  test('Clear listening history empties Recently played (#5, #6)', async ({ page }, testInfo) => {
    await signInIsolated(page, 'xs-clear-history', testInfo)
    await listenToOne(page)

    await page.getByTestId('masthead-queue').click() // the reader: Recently played (and its cache)
    await expect(page.getByTestId('queue-panel-recent')).toBeVisible()

    await navTo(page, 'profile')
    await page.getByTestId('profile-clear-history-open').click()
    await page.getByTestId('profile-clear-history-confirm').click()
    await expect(page.getByTestId('profile-clear-history-result')).toBeVisible()

    await page.getByTestId('masthead-queue').click()
    await expect(page.getByTestId('queue-panel-recent'), 'the cleared history is still listed').toHaveCount(0)
  })

  test('"Your trends" on a kept-alive Discover fill in after a follow (#8)', async ({ page }, testInfo) => {
    await signInIsolated(page, 'xs-mine-trends', testInfo)
    await navTo(page, 'catalog') // the reader: Discover, Trends on "mine" (the default)
    await expect(page.getByTestId('home-trending-scope')).toHaveAttribute('aria-pressed', 'true')
    await expect(page.getByTestId('discovery-mine-empty')).toBeVisible()

    // "Mine" is what you follow, save, or met in episodes you heard — a saved POSITION alone is not
    // in it. Follow a topic the fixture trends on, server-side (as from another device).
    await navTo(page, 'library')
    const r = await page.request.post('/api/app/interests/' + encodeURIComponent('topic:systems-thinking'))
    expect(r.ok()).toBeTruthy()

    await navTo(page, 'catalog')
    await expect(page.getByTestId('discovery-mine-empty'), 'Discover kept the trends from before the follow').toHaveCount(0)
  })
})
