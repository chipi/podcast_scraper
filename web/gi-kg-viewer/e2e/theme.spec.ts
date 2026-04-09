import { expect, test } from '@playwright/test'

test.describe('UXS theme tokens', () => {
  test('dark baseline uses dark canvas token', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'dark' })
    await page.goto('/')
    const canvas = await page.evaluate(() =>
      getComputedStyle(document.documentElement).getPropertyValue('--ps-canvas').trim().toLowerCase(),
    )
    expect(canvas).toBe('#111418')
  })

  test('light scheme switches canvas token', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'light' })
    await page.addInitScript(() => {
      localStorage.setItem('gi-kg-viewer-theme', 'light')
    })
    await page.goto('/')
    const canvas = await page.evaluate(() =>
      getComputedStyle(document.documentElement).getPropertyValue('--ps-canvas').trim().toLowerCase(),
    )
    expect(canvas).toBe('#f6f7f9')
  })
})
