import { expect, test } from '@playwright/test'

/**
 * Browse hub + standalone Topic/Person deep-links (#1261-6, #1261-9, #14). Real API +
 * committed corpus. Home surfaces a "Browse topics" / "Browse people" chip strip; each chip
 * deep-links into the unified Browse HUB on the matching tab (#14 folded the three standalone
 * index pages into one tabbed hub). Tapping a topic chip lands on the standalone Topic page.
 *
 * The hub replaces the mobile-hostile Cmd-K palette that was explicitly ruled out of the player.
 */

test('Home surfaces "Browse topics" / "Browse people" and each deep-links into the hub', async ({
  page,
}) => {
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
}) => {
  // `topic:risk-management` is a core committed-corpus topic (10 distinct-position shows —
  // perspectives.spec). Navigate to its standalone page directly: the Browse trending index is
  // temporal_velocity-backed, an enrichment the committed corpus deliberately does NOT ship, so
  // there is no topic chip to seed off in e2e (the chips DO render in prod). The page render itself —
  // EntityCardBody in variant='inline' — is what this spec proves.
  await page.goto('/topic/' + encodeURIComponent('topic:risk-management'))
  await expect(page.getByTestId('topic-view')).toBeVisible()
  await expect(page.getByText('Topic', { exact: true })).toBeVisible()
})
