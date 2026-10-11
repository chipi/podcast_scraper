import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Search folded into Discovery (operator 2026-09-20), then moved to Explore (operator 2026-10-11:
 * looking for something is Explore's job; Discover is being told what to hear next).
 *
 * Search stopped being a top-level destination. `/search` is unchanged — this is navigation, not
 * routing — so what needs covering is where you ENTER from and which tab reads as "you are here".
 *
 * Unit tests already pin the tab list and the ownership map. What only a browser can show is the
 * part that actually bit before: the masthead icons are hidden below the `sm` breakpoint by a
 * media query (the magnifier too, since 2026-09-30), and jsdom does not evaluate media queries. A mounted test passes either way — which
 * is exactly how #1588 (search unreachable from most of the app) happened the first time. These run
 * on `mobile-chrome` AND `desktop-chrome`, so the breakpoint is exercised from both sides.
 */

test('the phone tab bar has three destinations and Search is not one of them', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'sid-tabs', testInfo)
  await page.goto('/')

  await expect(page.locator('[data-testid="bottom-nav-search"]')).toHaveCount(0)
  for (const name of ['home', 'browse', 'explore', 'library']) {
    await expect(page.locator(`[data-testid="bottom-nav-${name}"]`)).toHaveCount(1)
  }
})

test('search is reachable from any screen, at any width (#1588)', async ({ page }, testInfo) => {
  await signInIsolated(page, 'sid-masthead', testInfo)

  // Deliberately NOT Home: #1588 was that search was unreachable from the catalogue, the player, the
  // library and show pages. Home has its own Ask box, so testing there would prove nothing.
  await page.goto('/library')
  const mast = page.getByTestId('masthead-search')
  if (testInfo.project.name === 'mobile-chrome') {
    // Phone: no magnifier in the header (2026-09-30, no room for it). The Explore tab is the way
    // in, and it is on this screen too.
    await expect(mast).toBeHidden()
    await page.getByTestId('bottom-nav-explore').click()
    await page.getByTestId('browse-search-input').fill('risk')
    await page.getByTestId('browse-search-submit').click()
  } else {
    await expect(mast).toBeVisible()
    await mast.click()
  }
  await expect(page).toHaveURL(/\/search/)
})

test('Explore carries the search box, and it runs the search', async ({ page }, testInfo) => {
  await signInIsolated(page, 'sid-box', testInfo)
  await page.goto('/explore')

  const box = page.getByTestId('browse-search-section')
  await expect(box).toBeVisible()

  await page.getByTestId('browse-search-input').fill('risk')
  await page.getByTestId('browse-search-submit').click()

  await expect(page).toHaveURL(/\/search\?q=risk/)
  // A real results page, not just a URL change: the scope control only renders once SearchView is
  // mounted with a query.
  await expect(page.getByTestId('search-scope')).toBeVisible()
})

test('the search box leads Explore, above Trends, and Discover has none', async ({ page }, testInfo) => {
  // Placement is the operator's instruction, so it is asserted by POSITION rather than presence:
  // "it renders somewhere on Explore" would stay green if it drifted to the bottom of the page.
  // On a phone the Explore tab IS the way into search, so it leads (operator 2026-10-11).
  await signInIsolated(page, 'sid-order', testInfo)
  await page.goto('/explore')
  const box = page.getByTestId('browse-search-section')
  const trends = page.getByTestId('discovery-explorer')
  for (const l of [box, trends]) await expect(l).toBeVisible()
  const y = async (l: typeof box) => (await l.boundingBox())!.y
  expect(await y(box)).toBeLessThan(await y(trends))

  await page.goto('/browse')
  await expect(page.getByTestId('quote-cards')).toBeVisible()
  await expect(page.getByTestId('browse-search-section')).toHaveCount(0)
})

test('the DESKTOP masthead agrees with the phone bar on /search', async ({ page }, testInfo) => {
  /*
   * The two navs are different components rendering one IA, and they disagreed: the bar used its
   * own ownership map, the masthead fell back to RouterLink's exact-active. On /search that lit
   * Discovery on a phone and Search on a desktop — the same URL answering "where am I" two ways
   * depending on window width (operator 2026-09-20: keep desktop in sync with mobile).
   *
   * Runs in BOTH projects. On mobile-chrome the masthead icons are `hidden … sm:flex`, so this
   * asserts what is true at that width instead of forcing a desktop-only expectation.
   */
  await signInIsolated(page, 'sid-desktop', testInfo)
  await page.goto('/search?q=risk')

  // By TESTID, not by label: a label matched nothing once, both projects fell to the else branch,
  // and the desktop assertions below never ran — 12/12 green while proving nothing. Found in review.
  const exploreIcon = page.getByTestId('masthead-explore')
  if (await exploreIcon.isVisible().catch(() => false)) {
    // Desktop width: Explore owns /search here too (2026-10-11), so its icon carries the marker...
    await expect(exploreIcon).toHaveAttribute('aria-current', 'page')
    // ...and the Search icon does NOT, even though it is the link to this very page.
    await expect(page.getByTestId('masthead-search')).not.toHaveAttribute('aria-current', 'page')
  } else {
    // Phone width: the icons are hidden, and the bottom bar carries the answer instead.
    await expect(page.locator('[data-testid="bottom-nav-explore"]')).toHaveAttribute(
      'aria-current',
      'page',
    )
  }
})

test('Explore stays lit on the search results page', async ({ page }, testInfo) => {
  /*
   * The operator's call, and it knowingly overrules BottomNav's "a wrong 'you are here' is worse
   * than none". Pinned here as well as in the unit test because this is the user-visible half: a
   * regression would silently return the bar to lighting nothing on /search, which would look like
   * a bug rather than a reverted decision.
   */
  await signInIsolated(page, 'sid-lit', testInfo)
  await page.goto('/search?q=risk')

  await expect(page.locator('[data-testid="bottom-nav-explore"]')).toHaveAttribute(
    'aria-current',
    'page',
  )
  await expect(page.locator('[data-testid="bottom-nav-home"]')).not.toHaveAttribute(
    'aria-current',
    'page',
  )
})
