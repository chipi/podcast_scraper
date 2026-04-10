import type { Page } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from './fixtures'

/** Shell `<h1>` product title; v2 lives in a child span (accessible name includes it). */
export const SHELL_HEADING_RE = /Podcast Intelligence Platform/i

/** Header nav (Digest / Library / Graph / Dashboard) — scope clicks to avoid substring clashes (e.g. “Open Library”, “Load into graph”). */
export function mainViewsNav(page: Page) {
  return page.getByRole('navigation', { name: 'Main views' })
}

/** Left rail: Corpus path vs API · Data (connection + data overview). */
export function leftPanelTabs(page: Page) {
  return page.getByRole('navigation', { name: 'Left panel tabs' })
}

/**
 * Offline graph load: force /api/health to fail so the "Choose files…" control
 * is shown on the **API · Data** tab, then load the CI fixture via the file picker.
 */
export async function loadGraphViaFilePicker(page: Page): Promise<void> {
  await page.route('**/api/health', async (route) => {
    await route.abort('failed')
  })

  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

  await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

  await leftPanelTabs(page).getByRole('button', { name: 'API · Data' }).click()

  const chooseBtn = page.getByRole('button', { name: /Choose files/i })
  await chooseBtn.waitFor({ state: 'visible', timeout: 30_000 })

  const fileInput = page.locator('input[type="file"]').first()
  await fileInput.setInputFiles(GI_SAMPLE_FIXTURE)

  await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
}
