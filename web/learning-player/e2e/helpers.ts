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

/**
 * Reveal the transcript when it's collapsed.
 *
 * The transcript is opt-in on mobile (a "Show transcript" toggle) so pressing
 * play doesn't jump the listener into it; on desktop it's the always-visible
 * side column and the toggle is hidden. This clicks the toggle only when it's
 * actually visible — a no-op on desktop — so transcript specs pass under both
 * Playwright projects (mobile-chrome + desktop-chrome).
 */
export async function openTranscript(page: Page): Promise<void> {
  const toggle = page.getByTestId('transcript-toggle')
  // Wait for the toggle to attach — it renders once segments load, on BOTH viewports
  // (lg:hidden on desktop). Then click only if it's actually visible (mobile); on desktop
  // it's attached-but-hidden and the transcript is already the side column, so this no-ops.
  await toggle.waitFor({ state: 'attached', timeout: 15_000 }).catch(() => {})
  if (await toggle.isVisible().catch(() => false)) {
    await toggle.click()
  }
}
