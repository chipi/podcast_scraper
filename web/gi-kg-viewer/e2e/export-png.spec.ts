import { expect, test } from '@playwright/test'
import { loadGraphViaFilePicker } from './helpers'

test.describe('Graph export', () => {
  test('Export PNG triggers download', async ({ page }) => {
    await loadGraphViaFilePicker(page)

    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('button', { name: 'Export PNG' }).click(),
    ])

    expect(download.suggestedFilename().toLowerCase()).toMatch(/\.png$/i)
  })
})
