import { expect, type Page, type TestInfo } from '@playwright/test'

/**
 * Sign in as an ISOLATED mock identity, unique per (spec, project). The mock OAuth provider honours
 * the `?as=` hint (dev/e2e only) and self-completes as `e2e-<hint>` — so parallel specs don't share
 * one mock user (which would race on the shared per-user files). `who` should be the spec's name.
 */
export async function signInIsolated(page: Page, who: string, testInfo: TestInfo): Promise<void> {
  const id = `${who}-${testInfo.project.name}`.toLowerCase().replace(/[^a-z0-9-]/g, '')
  await page.goto(`/api/app/auth/login?as=${encodeURIComponent(id)}`)
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()
}
