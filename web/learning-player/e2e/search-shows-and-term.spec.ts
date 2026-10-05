import { expect, test } from '@playwright/test'
import { navTo, signInIsolated } from './helpers'

/**
 * Search, end to end against the real API (operator 2026-10-04):
 *  - a SHOW is a search result, matched by every word in any order — "horizon long" finds
 *    "Long Horizon Notes", which the passage search alone never returned as a show;
 *  - a search started from Home lands on the results page with the box showing THAT term, even
 *    when the results page was already open (it is kept alive and used to keep the old one);
 *  - the results say what they are results for, in the count row.
 */
test('a show is found by its name, words in any order, and the count row names the term', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'search-shows', testInfo)
  await page.goto('/search?q=horizon%20long')
  const shows = page.getByTestId('search-shows')
  await expect(page.getByTestId('search-section-shows')).toBeVisible()
  await expect(shows.getByText('Long Horizon Notes')).toBeVisible()

  await page.goto('/search?q=risk%20management')
  await expect(page.getByText(/passages across \d+ episodes for “risk management”/)).toBeVisible()
})

test('a search from Home shows its own term in the results box, not the previous one', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'search-home-term', testInfo)
  // Open the results page first, so it is alive with an OLD term when Home's search lands on it.
  await page.goto('/search?q=memory')
  await expect(page.locator('input[type="search"]').first()).toHaveValue('memory')

  // In-app (navTo picks the nav this viewport shows), so the results page stays alive behind it.
  await navTo(page, 'home')
  const homeBox = page.getByTestId('home-search-input')
  await homeBox.fill('risk management')
  await page.getByTestId('home-search-submit').click()

  await expect(page).toHaveURL(/\/search\?q=risk/)
  await expect(page.locator('input[type="search"]').first()).toHaveValue('risk management')
})
