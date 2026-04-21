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

  test('/ from Explore mode switches back to Search and focuses query', async ({ page }) => {
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

    await page.getByTestId('left-panel-enter-explore').click()
    await expect(page.getByRole('heading', { name: /Explore & query/i })).toBeVisible()

    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('/')

    await expect(page.locator('#search-q')).toBeFocused()
    await expect(page.locator('#search-q')).toBeVisible()
  })

  test('Esc clears graph interaction on Graph tab (offline load)', async ({ page }) => {
    await loadGraphViaFilePicker(page)
    await dismissGraphGestureOverlayIfPresent(page)

    await page.locator('.graph-canvas').click({ position: { x: 120, y: 120 } })
    await page.keyboard.press('Escape')

    await expect(page.getByRole('button', { name: 'Fit' })).toBeVisible()
  })
})
