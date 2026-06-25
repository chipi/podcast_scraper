import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

/**
 * Home sections + corpus-search entry — REAL API, REAL fixture corpus, NO mocks. Asserts the
 * What's-new + Your-shows sections (two shows from the backend) and that the search entry
 * routes to /search. The fixture has no vector index, so search degrades gracefully (the
 * real results path is covered by the Docker stack-test against the airgapped index).
 */
test('Home shows sections; search routes to /search and degrades gracefully', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: "What's new" })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Your shows' })).toBeVisible()
  await expect(page.getByText('Memory Lab').first()).toBeVisible()
  await expect(page.getByText('Money Talk').first()).toBeVisible()

  const homeAxe = await new AxeBuilder({ page }).analyze()
  expect(homeAxe.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious')).toEqual([])

  // Search entry → /search (no index in the fixture → graceful message, not a broken panel).
  await page.getByLabel('Ask your library').fill('memory')
  await page.getByRole('button', { name: 'Search your library' }).first().click()
  await expect(page).toHaveURL(/\/search\?q=memory/)
  await expect(page.getByText('Search needs the library index.')).toBeVisible()
})
