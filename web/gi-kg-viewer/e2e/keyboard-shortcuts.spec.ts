import { expect, test } from '@playwright/test'
import { dismissGraphGestureOverlayIfPresent, loadGraphViaFilePicker, statusBarCorpusPathInput, mockSignIn } from './helpers'

test.describe('Keyboard shortcuts', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test('/ focuses semantic search when API is healthy', async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
        }),
      })
    })

    await page.goto('/')
    await page.locator('#search-q').waitFor({ state: 'visible', timeout: 30_000 })
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.locator('#search-q')).toBeEnabled({ timeout: 10_000 })

    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('/')

    await expect(page.locator('#search-q')).toBeFocused()
  })

  // Removed in Search v3 §S1 (Explore mode retired): the previous test
  // "/ from Explore mode switches back to Search and focuses query" is no
  // longer meaningful — there is no Explore mode to switch out of.
  // A "/" from any tab still focuses ``#search-q`` per the general shortcut
  // (covered by the earlier test in this describe block).

  test('Esc clears graph interaction on Graph tab (offline load)', async ({ page }) => {
    await loadGraphViaFilePicker(page)
    await dismissGraphGestureOverlayIfPresent(page)

    await page.locator('.graph-canvas').click({ position: { x: 120, y: 120 } })
    await page.keyboard.press('Escape')

    await expect(page.getByRole('button', { name: 'Fit' })).toBeVisible()
  })
})
