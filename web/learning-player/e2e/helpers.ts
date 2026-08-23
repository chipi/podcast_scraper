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


/**
 * Navigate in-app to a primary destination, whichever nav is on screen.
 *
 * The app has two navs by design: a bottom tab bar below `sm`, and header icon links at and above
 * it — each hidden at the other's widths. A spec that hard-codes one only runs on one project, and
 * `page.goto` is not a substitute: it is a full page load, which tears down the SPA and stops
 * audio, so it cannot test anything about client-side navigation.
 *
 * Browse has no tab (it is a corpus index, not a daily destination), so on mobile it is reached
 * from Home's "Browse all →" link instead.
 */
export async function navTo(
  page: Page,
  dest: 'home' | 'search' | 'library' | 'profile' | 'catalog',
): Promise<void> {
  if (dest === 'catalog') {
    const tab = page.getByTestId('bottom-nav-home')
    if (await tab.isVisible().catch(() => false)) {
      await tab.click()
      await page.getByRole('link', { name: /Browse all/i }).first().click()
      return
    }
    await page.locator('header').getByRole('link', { name: 'Browse' }).click()
    return
  }

  const tab = page.getByTestId(`bottom-nav-${dest}`)
  if (await tab.isVisible().catch(() => false)) {
    await tab.click()
    return
  }
  const LABELS: Record<string, string> = {
    home: 'Podcast Learning Player',
    search: 'Search',
    library: 'Library',
    profile: 'Profile',
  }
  await page.locator('header').getByRole('link', { name: LABELS[dest] }).click()
}
