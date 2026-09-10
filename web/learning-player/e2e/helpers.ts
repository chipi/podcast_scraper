import { expect, type Page, type TestInfo } from '@playwright/test'

/**
 * Sign in as an ISOLATED mock identity, unique per (spec, project). The mock OAuth provider honours
 * the `?as=` hint (dev/e2e only) and self-completes as `e2e-<hint>` — so parallel specs don't share
 * one mock user (which would race on the shared per-user files). `who` should be the spec's name.
 */
/**
 * The signed-in signal, in ONE place (#1962).
 *
 * Every spec used to assert `Sign out` is visible in the masthead. That button moved to Profile —
 * the top-right of a mobile app should hold the most-used action, and it held the least-used one.
 * The remaining masthead signal is the ABSENCE of the sign-in link, which is a real property of
 * the shell rather than a test-only hook.
 *
 * Centralised so the next time this moves it is one edit, not eleven.
 */
export async function expectSignedIn(page: Page): Promise<void> {
  await expect(page.getByRole('link', { name: 'Sign in' })).toHaveCount(0)
  // Signed-in POSITIVE signal: a reachable Profile link. Deliberately NOT the bottom-nav testid —
  // the tab bar is `sm:hidden` and the header icons are `hidden sm:flex`, so exactly one of the two
  // exists-and-is-visible at any width, and pinning the mobile one made every desktop-chrome spec
  // fail against an element that was present but hidden. `:visible` picks whichever the current
  // project renders, so this holds under both.
  await expect(page.locator('a[href="/profile"]:visible').first()).toBeVisible()
}

export async function signInIsolated(page: Page, who: string, testInfo: TestInfo): Promise<void> {
  const id = `${who}-${testInfo.project.name}`.toLowerCase().replace(/[^a-z0-9-]/g, '')
  await page.goto(`/api/app/auth/login?as=${encodeURIComponent(id)}`)
  await expectSignedIn(page)
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

  // Profile moved out of the bottom nav to the masthead avatar (2026-09-09) — reach it there.
  if (dest === 'profile') {
    await page.getByTestId('header-profile').click()
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
