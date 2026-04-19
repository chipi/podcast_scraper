import { expect, test } from '@playwright/test'
import {
  dismissGraphGestureOverlayIfPresent,
  loadGraphViaFilePicker,
  statusBarCorpusPathInput,
} from './helpers'

test.describe('Keyboard shortcuts', () => {
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

  test('Esc clears graph interaction on Graph tab (offline load)', async ({ page }) => {
    await loadGraphViaFilePicker(page)
    await dismissGraphGestureOverlayIfPresent(page)

    await page.locator('.graph-canvas').click({ position: { x: 120, y: 120 } })
    await page.keyboard.press('Escape')

    await expect(page.getByRole('button', { name: 'Fit' })).toBeVisible()
  })
})
