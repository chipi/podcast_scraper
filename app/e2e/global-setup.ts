import { chromium, type FullConfig } from '@playwright/test'

/**
 * Warm the mock-OAuth user once, before the parallel specs run. All e2e specs sign in as the SAME
 * mock identity (the mock provider returns a fixed subject); without this, several specs racing on
 * the FIRST login concurrently can collide while the per-user store is being created. A single
 * upfront login creates that record so every spec's sign-in is a returning user (no create race).
 */
export default async function globalSetup(config: FullConfig): Promise<void> {
  const baseURL = config.projects[0]?.use?.baseURL ?? 'http://127.0.0.1:4174'
  const browser = await chromium.launch()
  try {
    const page = await browser.newPage({ baseURL })
    await page.goto('/')
    await page.getByRole('link', { name: 'Sign in' }).click()
    await page.getByRole('button', { name: 'Sign in' }).click()
    await page.getByRole('button', { name: 'Sign out' }).waitFor({ state: 'visible' })
  } finally {
    await browser.close()
  }
}
