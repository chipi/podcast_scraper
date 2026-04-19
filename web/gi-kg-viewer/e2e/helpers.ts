import type { Page } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from './fixtures'

/** Shell `<h1>` product title; v2 lives in a child span (accessible name includes it). */
export const SHELL_HEADING_RE = /Podcast Intelligence Platform/i

/** Header nav (Digest / Library / Graph / Dashboard) — scope clicks to avoid substring clashes (e.g. main-tab Library vs “Load into graph”). */
export function mainViewsNav(page: Page) {
  return page.getByRole('navigation', { name: 'Main views' })
}

/**
 * **Dashboard** tab: **Corpus artifacts** workspace (`data-testid="corpus-data-workspace"`)
 * — artifacts list, API health, and **Data** overview (same content as the former left **API · Data** tab).
 */
export async function openCorpusDataWorkspace(page: Page): Promise<void> {
  await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
  await page.getByTestId('corpus-data-workspace').waitFor({ state: 'visible' })
}

/** Bottom status bar corpus path field (`data-testid="status-bar-corpus-path"`). */
export function statusBarCorpusPathInput(page: Page) {
  return page.getByTestId('status-bar-corpus-path')
}

/**
 * Offline graph load: force /api/health to fail, open **Graph**, then load the CI
 * fixture via the status bar **Files** / hidden file input.
 */
/**
 * Closes the first-run graph gesture overlay when it is visible so pointer and
 * keyboard tests can reach Cytoscape (overlay is above `.graph-canvas`).
 */
export async function dismissGraphGestureOverlayIfPresent(page: Page): Promise<void> {
  const btn = page.getByTestId('graph-gesture-overlay-dismiss')
  try {
    await btn.waitFor({ state: 'visible', timeout: 3000 })
  } catch {
    return
  }
  await btn.click()
}

export async function loadGraphViaFilePicker(page: Page): Promise<void> {
  await page.route('**/api/health', async (route) => {
    await route.abort('failed')
  })

  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

  await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

  const fileInput = page.getByTestId('status-bar-local-file-input')
  await fileInput.setInputFiles(GI_SAMPLE_FIXTURE)

  await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
}
