import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Browse hub + standalone Topic/Person deep-links (#1261-6, #1261-9, #14). Real API +
 * committed corpus. Home surfaces a compact "Discover" strip (Topics · Storylines · People); each
 * chip deep-links into Browse's Trends section on the matching kind (operator 2026-09-14, renamed
 * from the old "Browse topics/people" links). Tapping a topic chip lands on the standalone Topic page.
 *
 * The hub replaces the mobile-hostile Cmd-K palette that was explicitly ruled out of the player.
 *
 * RFC-120: all routes below are login-first; each test signs in.
 */

test('Home surfaces the compact "Discover" strip and each chip deep-links into the Browse Trends section', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'browse-home-nav', testInfo)
  await page.goto('/')
  const nav = page.getByTestId('home-browse-nav')
  await expect(nav).toBeVisible()

  // Renamed from "Browse topics/people" to a compact Discover strip (operator 2026-09-14): three
  // chips — Topics · Storylines · People. They used to open a standalone /trends page that was a
  // thinner copy of Browse's own Trends section; that page is deleted and the chips deep-link into
  // the section itself (operator 2026-09-18).
  await expect(nav.getByTestId('home-discover-topics')).toBeVisible()
  await expect(nav.getByTestId('home-discover-storylines')).toBeVisible()
  await expect(nav.getByTestId('home-discover-people')).toBeVisible()

  await nav.getByTestId('home-discover-topics').click()
  await expect(page).toHaveURL(/\/browse\?trends=topic/)
  await expect(page.getByTestId('browse-view')).toBeVisible()
  // The KIND must be selected, not merely the page reached — the whole point of the deep link.
  await expect(page.getByTestId('discovery-tab-topic')).toHaveAttribute('aria-selected', 'true')

  await page.goto('/')
  await page.getByTestId('home-browse-nav').getByTestId('home-discover-people').click()
  await expect(page).toHaveURL(/\/browse\?trends=person/)
  await expect(page.getByTestId('discovery-tab-person')).toHaveAttribute('aria-selected', 'true')
})

test('the standalone /topic/:id page renders the topic card body (EntityCardBody inline mode)', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'topic-page-inline', testInfo)
  // `topic:risk-management` is a core committed-corpus topic (10 distinct-position shows —
  // perspectives.spec). Navigate to its standalone page directly: the Browse trending index is
  // temporal_velocity-backed, an enrichment the committed corpus deliberately does NOT ship, so
  // there is no topic chip to seed off in e2e (the chips DO render in prod). The page render itself —
  // EntityCardBody in variant='inline' — is what this spec proves.
  await page.goto('/topic/' + encodeURIComponent('topic:risk-management'))
  await expect(page.getByTestId('topic-view')).toBeVisible()
  await expect(page.getByText('Topic', { exact: true })).toBeVisible()
})

test('the trend-window selector defaults to 3M and switches (RFC-103 R2)', async ({ page }, testInfo) => {
  await signInIsolated(page, 'trend-window-3m', testInfo)
  // The trend-window control lives inside DiscoveryList, rendered on Home's discovery section.
  // The committed corpus ships no temporal_velocity, so rows may be empty — but the window
  // control always renders (so an empty window can be switched away from). Assert the control's
  // default + that a pick updates the selection; the refetch itself is covered by the unit tests.
  await page.goto('/')
  await expect(page.getByTestId('home-discovery')).toBeVisible()
  const section = page.getByTestId('home-discovery')
  // `aria-checked`, not `aria-selected` (#1594 item 7): the window selector is a radiogroup.
  // It re-queries the rail its PARENT owns and switches no panel, so `role="tab"` was promising a
  // panel that never existed.
  const tabs = section.getByTestId('trend-window-tabs')
  await expect(tabs).toBeVisible()
  await expect(section.getByTestId('trend-window-3m')).toHaveAttribute('aria-checked', 'true')

  await section.getByTestId('trend-window-6m').click()
  await expect(section.getByTestId('trend-window-6m')).toHaveAttribute('aria-checked', 'true')
  await expect(section.getByTestId('trend-window-3m')).toHaveAttribute('aria-checked', 'false')
})

/**
 * The catalogue is grouped by WHEN, and only when time is the order (#1978).
 *
 * Measured problem, not an assumed one: 29 structurally identical rows down a 4,929px page with
 * nothing to break them — "a spreadsheet with pictures". The critic's own alternative was "a
 * divider every six rows", which breaks monotony while meaning nothing. These headings carry
 * information instead.
 *
 * The half of the contract worth guarding hardest is the NEGATIVE one: under a title sort, or with
 * a search term active, a "This week" heading over an alphabetical list would be a lie. A test that
 * only checked headings appear would let that regression through.
 */
test('the catalogue groups by time, and stops when time is not the order', async ({ page }, testInfo) => {
  await signInIsolated(page, 'catalogue-time-groups', testInfo)
  await page.goto('/browse')
  await page.waitForLoadState('networkidle')

  const headings = page.locator('h2.lp-kicker')
  const grouped = await headings.allInnerTexts()
  expect(grouped.length, 'a newest-first catalogue should carry at least one time band').toBeGreaterThan(0)
  // Bands appear only when populated — an empty "This week" would be furniture, not information.
  for (const h of grouped) {
    expect(h).toMatch(/this week|earlier this month|earlier this year|before that|undated/i)
  }

  // Sorting by A–Z makes the time order untrue, so the bands must go. The sort control is a
  // ToolbarMenu now (not a native <select>): open it, pick the A–Z option.
  const sortBtn = page.getByTestId('list-toolbar-sort')
  if (await sortBtn.isVisible().catch(() => false)) {
    await sortBtn.click()
    await page.getByTestId('list-toolbar-sort-opt-az').click()
    await page.waitForTimeout(300)
    await expect(
      page.locator('h2.lp-kicker'),
      'time bands over an alphabetical list would be a lie',
    ).toHaveCount(0)
  }
})
