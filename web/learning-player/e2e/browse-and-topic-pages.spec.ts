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

test('a topic chip opens the standalone /topic/:id page (EntityCardBody inline mode)', async ({
  page,
}) => {
  // The standalone topics index still routes; it renders the trending topic chips (TrendingSparkChips
  // → `trend-spark-row`, a button that router.push-es to /topic/:id — no anchor to read an href off).
  await page.goto('/browse/topics')
  const chip = page.getByTestId('trend-spark-row').first()
  await expect(chip).toBeVisible()
  await chip.click()
  await expect(page).toHaveURL(/\/topic\//)
  // EntityCardBody renders in variant='inline' — the "Topic" kicker plus the topic label.
  await expect(page.getByTestId('topic-view')).toBeVisible()
  await expect(page.getByText('Topic', { exact: true })).toBeVisible()
})
