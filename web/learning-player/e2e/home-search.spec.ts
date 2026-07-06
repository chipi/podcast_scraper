import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

/**
 * Home sections + corpus-search entry — REAL API over the COMMITTED validation corpus, NO mocks.
 * Asserts the What's-new + All-shows sections (real shows from the backend) and that the search
 * entry routes to /search. The committed corpus has no vector index, so search degrades
 * gracefully (the real results path is covered by the Docker stack-test against the airgapped
 * index).
 *
 * Shows come from the committed corpus: "Long Horizon Notes" (p05), "Practical Systems" (p02),
 * "Below the Surface" (p03).
 */
test('Home shows sections; search routes to /search and degrades gracefully', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: "What's new" })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'All shows' })).toBeVisible()
  await expect(page.getByText('Long Horizon Notes').first()).toBeVisible()
  await expect(page.getByText('Below the Surface').first()).toBeVisible()

  const homeAxe = await new AxeBuilder({ page }).analyze()
  expect(homeAxe.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious')).toEqual([])

  // Search entry → /search (no index in the committed corpus → graceful message, not a broken panel).
  await page.getByLabel('Ask across every episode').fill('investing')
  await page.getByRole('button', { name: 'Search', exact: true }).first().click()
  await expect(page).toHaveURL(/\/search\?q=investing/)
  await expect(page.getByText('Search needs the library index.')).toBeVisible()
})
