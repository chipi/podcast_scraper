import { expect, test } from '@playwright/test'
import { dismissGraphGestureOverlayIfPresent, loadGraphViaFilePicker, statusBarCorpusPathInput, mockSignIn } from './helpers'

test.describe('Keyboard shortcuts', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test('/ opens the command palette (Search v3 §S3 + §S4-shell)', async ({ page }) => {
    await page.route('**/api/health**', async (route) => {
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
    await statusBarCorpusPathInput(page).fill('/mock/corpus')

    // Take focus off any input so `/` isn't captured as a literal keystroke.
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('/')

    await expect(page.getByTestId('command-palette')).toBeVisible()
    await expect(page.getByTestId('command-palette-input')).toBeFocused()
  })

  // Removed in Search v3 §S1 (Explore mode retired) and superseded by §S3
  // (palette): `/` no longer focuses a launcher — it summons the shell-wide
  // command palette. The compact SearchPanel launcher itself retired in
  // §S4-shell (search only lives in the main-window Search tab).

  test('Esc clears graph interaction on Graph tab (offline load)', async ({ page }) => {
    await loadGraphViaFilePicker(page)
    await dismissGraphGestureOverlayIfPresent(page)

    await page.locator('.graph-canvas').click({ position: { x: 120, y: 120 } })
    await page.keyboard.press('Escape')

    await expect(page.getByRole('button', { name: 'Fit' })).toBeVisible()
  })
})
