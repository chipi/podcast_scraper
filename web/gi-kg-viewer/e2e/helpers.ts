import type { Page } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from './fixtures'

/**
 * Offline graph load: force /api/health to fail so the "Choose files…" control
 * is shown on the API tab, then load the CI fixture via the file picker.
 */
export async function loadGraphViaFilePicker(page: Page): Promise<void> {
  await page.route('**/api/health', async (route) => {
    await route.abort('failed')
  })

  await page.goto('/')
  await page.getByRole('heading', { name: /GI \/ KG Viewer/i }).waitFor()

  await page.getByRole('button', { name: 'API' }).click()

  const chooseBtn = page.getByRole('button', { name: /Choose files/i })
  await chooseBtn.waitFor({ state: 'visible', timeout: 30_000 })

  const fileInput = page.locator('input[type="file"]').first()
  await fileInput.setInputFiles(GI_SAMPLE_FIXTURE)

  await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
}
