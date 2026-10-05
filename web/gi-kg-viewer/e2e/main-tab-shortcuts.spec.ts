import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * UXS-001 global shortcuts (`useViewerKeyboard`): 1–5 switch the main tab when focus is not in
 * an editable control; Ctrl+K summons the command palette from anywhere, inputs included.
 * Admin, because 5 is the Dashboard and only admins have it.
 */

async function openDigest(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
  await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
  await expect(page.getByTestId('digest-root')).toBeVisible()
  // Move focus off the corpus path input so the digits are shortcuts, not typed text.
  await page.locator('body').click({ position: { x: 5, y: 5 } })
}

test.describe('Main-tab keyboard shortcuts (UXS-001)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin', { liveApi: true })
  })

  test('2 / 3 / 4 / 5 / 1 switch Library / Search / Graph / Dashboard / Digest', async ({
    page,
  }) => {
    await openDigest(page)

    await page.keyboard.press('2')
    await expect(page.getByTestId('library-root')).toBeVisible()
    await expect(page.getByTestId('digest-root')).toBeHidden()

    await page.keyboard.press('3')
    await expect(page.getByTestId('search-workspace')).toBeVisible()

    // Focus lands in the search box on that tab; leave it so digits stay shortcuts.
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('4')
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()

    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('5')
    await expect(page.getByTestId('briefing-card')).toBeVisible()

    await page.keyboard.press('1')
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await expect(page.getByTestId('briefing-card')).toBeHidden()
  })

  test('a digit typed into the search box is text, not a tab switch', async ({ page }) => {
    await openDigest(page)
    await page.keyboard.press('3')
    const q = page.locator('#search-q')
    await expect(page.getByTestId('search-workspace')).toBeVisible()
    await q.click()
    await q.fill('')
    await page.keyboard.type('2024')
    await expect(q).toHaveValue('2024')
    await expect(page.getByTestId('search-workspace')).toBeVisible()
    await expect(page.getByTestId('library-root')).toBeHidden()
    await expect(page.getByTestId('graph-tab-panel')).toBeHidden()
  })

  test('Control+K opens the command palette, even from inside an input', async ({ page }) => {
    await openDigest(page)
    await page.keyboard.press('Control+k')
    await expect(page.getByTestId('command-palette')).toBeVisible()
    await expect(page.getByTestId('command-palette-input')).toBeFocused()

    await page.keyboard.press('Escape')
    await expect(page.getByTestId('command-palette')).toBeHidden()

    await statusBarCorpusPathInput(page).click()
    await page.keyboard.press('Control+k')
    await expect(page.getByTestId('command-palette')).toBeVisible()
  })
})
