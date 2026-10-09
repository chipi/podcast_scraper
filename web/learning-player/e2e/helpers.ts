import { expect, type Locator, type Page, type TestInfo } from '@playwright/test'

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

/**
 * The mock login keeps only the first 32 characters of `?as=` (`_safe_hint` on the server), so two
 * long ids that differ only at the end were ONE account: every `--repeat-each` copy of
 * "trends-mine-fresh-desktop-chrome-r1" signed into the first copy's account and saw its state
 * (2026-10-08). A long id keeps a readable prefix and ends in a hash of the whole id.
 */
function accountHint(raw: string): string {
  const id = raw.toLowerCase().replace(/[^a-z0-9-]/g, '')
  if (id.length <= 32) return id
  let h = 0x811c9dc5
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 0x01000193) >>> 0
  return `${id.slice(0, 23)}-${h.toString(16).padStart(8, '0')}`
}

export async function signInIsolated(page: Page, who: string, testInfo: TestInfo): Promise<void> {
  // Per REPEAT too. `--repeat-each` runs copies of a test in parallel, and copies sharing one
  // account read each other's state — one copy's Save showed up as "Saved ✓" in another
  // (2026-10-04). A normal run has repeatEachIndex 0 and keeps its existing account id.
  const repeat = testInfo.repeatEachIndex > 0 ? `-r${testInfo.repeatEachIndex}` : ''
  const id = accountHint(`${who}-${testInfo.project.name}${repeat}`)
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
  await tapTo(page, dest)
  // ARRIVE before returning (2026-10-09). A click only starts a client-side navigation; a caller
  // that acts next — a write, another `navTo` — could otherwise run while the PREVIOUS page is still
  // on screen. That made a cross-surface test "leave" Discover without ever leaving it, so the tab
  // was never deactivated and the return it meant to test never happened.
  const at: Record<typeof dest, (u: URL) => boolean> = {
    home: (u) => u.pathname === '/',
    search: (u) => u.pathname.startsWith('/search'),
    library: (u) => u.pathname.startsWith('/library'),
    profile: (u) => u.pathname.startsWith('/profile'),
    catalog: (u) => u.pathname.startsWith('/browse') || u.pathname.startsWith('/catalog'),
  }
  await page.waitForURL(at[dest])
}

async function tapTo(
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
    // The header nav link was renamed Browse → Discover (nav.browse, operator 2026-09-14).
    await page.locator('header').getByRole('link', { name: 'Discover' }).click()
    return
  }

  // Search has no phone nav entry of its own: no tab (2026-09-20) and no header magnifier since
  // 2026-09-30 (no room for it). A phone reaches it the way a user does — Discover's search box —
  // which is still an IN-APP navigation, as audio-continuity needs. Desktop keeps the magnifier.
  if (dest === 'search') {
    const discover = page.getByTestId('bottom-nav-browse')
    if (await discover.isVisible().catch(() => false)) {
      await discover.click()
      await page.getByTestId('browse-search-input').fill('risk')
      await page.getByTestId('browse-search-submit').click()
      await page.waitForURL(/\/search/)
      return
    }
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
  // `home` is a REGEX because the header's home link is the brand lockup: its
  // accessible name is the tagline plus `app.title`, not a bare product name.
  // It read 'Podcast Learning Player' until now — a name the #2118 rename left
  // behind, and one `getByRole({ name })` could never match, since that option
  // is an exact match on the normalised accessible name. Only reachable when
  // the bottom nav is hidden, which is why it never went red.
  const LABELS: Record<string, string | RegExp> = {
    home: /Close Listening/,
    search: 'Search',
    library: 'Library',
    profile: 'Profile',
  }
  await page.locator('header').getByRole('link', { name: LABELS[dest] }).click()
}

/**
 * Tap `el` and return its top ON SCREEN AT THE TAP, read by a capture-phase click listener before
 * any app handler runs. A measurement taken before `click()` is not where the reader tapped: content
 * still loading above can move the control in between (measured 2026-10-04: 517px when measured,
 * 680px when tapped). The Back-lands-where-you-left contract is about where it was TAPPED.
 */
export async function tapAndRecordTop(el: Locator): Promise<number> {
  await el.page().evaluate(() => {
    const w = window as unknown as { __tapTop?: number }
    delete w.__tapTop
    window.addEventListener(
      'click',
      (e) => {
        const c = (e.target as Element).closest('button, a, [role="button"]') ?? (e.target as Element)
        w.__tapTop = c.getBoundingClientRect().top
      },
      { capture: true, once: true },
    )
  })
  await el.click()
  const top = await el.page().evaluate(() => (window as unknown as { __tapTop?: number }).__tapTop)
  expect(top, 'the tap never reached the page').not.toBeUndefined()
  return top as number
}

/**
 * Switch Discover to everyone's (operator 2026-10-07: "Mine" is the default, and a fresh test
 * account has no world of its own, so Mine is empty). Through the real page-level switch in
 * Discover's header (2026-10-09: one switch for trending shows, Trends and search), not a stored
 * preference, and only when it is on — so a spec reads the corpus-wide lists it is written about.
 */
export async function showEveryonesTrends(page: Page): Promise<void> {
  const toggle = page.getByTestId('discover-scope')
  await expect(toggle).toBeVisible()
  // The lens resolves from the synced preferences after mount; wait for it to settle on "mine".
  await expect(toggle).toHaveAttribute('aria-pressed', 'true')
  await toggle.click()
  await expect(toggle).toHaveAttribute('aria-pressed', 'false')
}

/** Play an episode the way the player does: two saves, two minutes apart in position. */
export async function listenToOne(page: Page): Promise<string> {
  const resp = await page.request.get('/api/app/episodes?page_size=1')
  const slug = ((await resp.json()).items as Array<{ slug: string }>)[0].slug
  const tz = -new Date().getTimezoneOffset()
  for (const position_seconds of [10, 130]) {
    const r = await page.request.put(`/api/app/playback/${slug}`, { data: { position_seconds, tz_offset_minutes: tz } })
    expect(r.ok()).toBeTruthy()
  }
  return slug
}
