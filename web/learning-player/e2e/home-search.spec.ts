import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

/**
 * Home sections + corpus-search entry — REAL API over the COMMITTED validation corpus, NO mocks.
 * Asserts the What's-new + All-shows sections (real shows from the backend) and that the search
 * entry routes to /search and returns REAL grounded results against the two-tier index that
 * e2e/globalSetup.ts builds for the committed corpus.
 *
 * Shows come from the committed corpus: "Long Horizon Notes" (p05), "Practical Systems" (p02),
 * "Below the Surface" (p03).
 */
test('Home shows sections; search routes to /search and returns grounded results', async ({
  page,
}) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: "What's new" })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'All shows' })).toBeVisible()
  await expect(page.getByText('Long Horizon Notes').first()).toBeVisible()
  await expect(page.getByText('Below the Surface').first()).toBeVisible()

  const homeAxe = await new AxeBuilder({ page }).analyze()
  expect(homeAxe.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious')).toEqual(
    [],
  )

  // Search entry → /search → real grounded results against the two-tier index (globalSetup builds
  // it; the warmup.setup project warms the serve's embedding model, so the summary line — which
  // renders only when results exist — is deterministic).
  await page.getByLabel('Ask across every episode').fill('investing')
  await page.getByRole('button', { name: 'Search', exact: true }).first().click()
  await expect(page).toHaveURL(/\/search\?q=investing/)
  await expect(page.getByText(/\d+ passages across \d+ episodes/)).toBeVisible()
})
