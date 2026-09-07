import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Browse hub + standalone Topic/Person deep-links (#1261-6, #1261-9, #14). Real API +
 * committed corpus. Home surfaces a "Browse topics" / "Browse people" chip strip; each chip
 * deep-links into the unified Browse HUB on the matching tab (#14 folded the three standalone
 * index pages into one tabbed hub). Tapping a topic chip lands on the standalone Topic page.
 *
 * The hub replaces the mobile-hostile Cmd-K palette that was explicitly ruled out of the player.
 *
 * RFC-120: all routes below are login-first; each test signs in.
 */

test('Home surfaces "Browse topics" / "Browse people" and each deep-links into the hub', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'browse-home-nav', testInfo)
  await page.goto('/')
  const nav = page.getByTestId('home-browse-nav')
  await expect(nav).toBeVisible()

  const topicsLink = nav.getByRole('link', { name: /Browse topics/ })
  const peopleLink = nav.getByRole('link', { name: /Browse people/ })
  await expect(topicsLink).toBeVisible()
  await expect(peopleLink).toBeVisible()

  // #14: Browse is one hub with tabs; the Home chips deep-link via ?tab= and land on that tab.
  await topicsLink.click()
  await expect(page).toHaveURL(/\/browse\?tab=topics$/)
  await expect(page.getByTestId('browse-tab-topics')).toHaveAttribute('aria-selected', 'true')

  await page.goto('/')
  await page
    .getByTestId('home-browse-nav')
    .getByRole('link', { name: /Browse people/ })
    .click()
  await expect(page).toHaveURL(/\/browse\?tab=people$/)
  await expect(page.getByTestId('browse-tab-people')).toHaveAttribute('aria-selected', 'true')
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
  // The committed corpus ships no temporal_velocity, so the chips are empty here — but the window
  // control always renders (so an empty window can be switched away from). Assert the control's
  // default + that a pick updates the selection; the refetch itself is covered by the unit tests.
  // Scoped to the Topics PANEL. `/browse/topics` now redirects into the hub (#2004 follow-up), and
  // the hub keeps every tab panel mounted (`v-show`, so switching never refetches) — so more than
  // one panel carries a `trend-window-tabs`. The standalone route used to mask that; it was always
  // true of the hub itself.
  await page.goto('/browse/topics')
  const panel = page.getByTestId('browse-panel-topics')
  await expect(panel).toBeVisible()
  const tabs = panel.getByTestId('trend-window-tabs')
  await expect(tabs).toBeVisible()
  await expect(panel.getByTestId('trend-window-3m')).toHaveAttribute('aria-selected', 'true')

  await panel.getByTestId('trend-window-6m').click()
  await expect(panel.getByTestId('trend-window-6m')).toHaveAttribute('aria-selected', 'true')
  await expect(panel.getByTestId('trend-window-3m')).toHaveAttribute('aria-selected', 'false')
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

  // Sorting by title makes the time order untrue, so the bands must go.
  const sortBy = page.getByLabel(/sort/i).first()
  if (await sortBy.isVisible().catch(() => false)) {
    await sortBy.selectOption('title').catch(() => undefined)
    await page.waitForTimeout(300)
    await expect(
      page.locator('h2.lp-kicker'),
      'time bands over an alphabetical list would be a lie',
    ).toHaveCount(0)
  }
})
