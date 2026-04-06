import { expect, test } from '@playwright/test'
import { loadGraphViaFilePicker } from './helpers'

test.describe('Offline graph (file picker)', () => {
  test('loads CI GI fixture and shows graph toolbar', async ({ page }) => {
    await loadGraphViaFilePicker(page)

    await expect(page.getByRole('button', { name: 'Fit' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Re-layout' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Export PNG' })).toBeVisible()
    await expect(page.locator('.graph-canvas')).toBeVisible()
  })
})
