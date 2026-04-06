import { expect, test } from '@playwright/test'
import { loadGraphViaFilePicker } from './helpers'

test.describe('Dashboard tab', () => {
  test('shows corpus overview after graph is loaded', async ({ page }) => {
    await loadGraphViaFilePicker(page)

    await page.getByRole('button', { name: 'Dashboard' }).click()

    await expect(page.getByRole('heading', { name: 'Corpus overview' })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Vector index' })).toBeVisible()
  })
})
