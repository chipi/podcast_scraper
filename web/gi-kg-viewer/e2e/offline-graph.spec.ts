import { expect, test } from '@playwright/test'
import { loadGraphViaFilePicker } from './helpers'

test.describe('Offline graph (file picker)', () => {
  test('loads CI GI fixture and shows graph toolbar', async ({ page }) => {
    await loadGraphViaFilePicker(page)

    await expect(page.getByRole('button', { name: 'Fit' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Re-layout' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Export PNG' })).toBeVisible()
    await expect(page.locator('.graph-canvas')).toBeVisible()

    await expect(page.getByRole('button', { name: 'Zoom out' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Zoom in' })).toBeVisible()
    await expect(page.getByRole('button', { name: '100%' })).toBeVisible()
    await expect(page.getByTestId('graph-gesture-overlay')).toBeVisible()
    await expect(page.getByTestId('graph-layout-cycle')).toBeVisible()
    await expect(page.getByTestId('graph-minimap-toggle')).toBeVisible()

    await page.getByTestId('graph-toolbar-more-filters').click()
    await expect(page.getByTestId('graph-filters-popover')).toBeVisible()
    await expect(page.getByRole('button', { name: /^0 \(\d+\)$/ })).toBeVisible()
  })
})
