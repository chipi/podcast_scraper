import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * News in search (UXS-019), against the live server over the app validation corpus's invented news
 * items (tests/fixtures/app-validation-corpus/build_news_items.py): the Type chip, the news row and
 * the article card. The server reads the fixture's publisher registry (playwright.config.ts).
 *
 * Every article body in the fixture holds NEWS-BODY-MARKER; no page may show it (ADR-179).
 */

async function openSearch(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
  await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
  await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
}

async function submit(page: Page, query: string): Promise<void> {
  await page.locator('#search-q').fill(query)
  await page
    .locator('section')
    .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
    .getByRole('button', { name: 'Search', exact: true })
    .click()
}

async function chooseType(page: Page, key: 'podcasts' | 'news' | 'both'): Promise<void> {
  await page.getByTestId('search-chip-type').click()
  await page.getByTestId(`search-type-option-${key}`).click()
}

test.describe('News in search (UXS-019, live)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('a podcast search shows no news; the Type chip brings news in', async ({ page }) => {
    await openSearch(page)
    await expect(page.getByTestId('search-chip-type')).toHaveText('Type ▾')
    await submit(page, 'risk management')
    await expect(page.getByTestId('search-result-tier').first()).toBeVisible({ timeout: 30_000 })
    await expect(page.getByTestId('search-result-news')).toHaveCount(0)

    await chooseType(page, 'news')
    await expect(page.getByTestId('search-chip-type')).toHaveText('Type: News ▾')
    const rows = page.getByTestId('search-result-news')
    await expect(rows.first()).toBeVisible({ timeout: 30_000 })
    await expect(page.getByTestId('search-result-tier')).toHaveCount(0)
    await expect(page.getByText('WITHHELD-ITEM-TITLE')).toHaveCount(0)
    await expect(page.locator('body')).not.toContainText('NEWS-BODY-MARKER')

    await chooseType(page, 'both')
    await expect(page.getByTestId('search-chip-type')).toHaveText('Type: Podcasts and news ▾')
    await expect(page.getByTestId('search-result-news').first()).toBeVisible({ timeout: 30_000 })
    await expect(page.getByTestId('search-result-tier').first()).toBeVisible()
  })

  test('the news row shows the item and its title opens the article card', async ({ page }) => {
    await openSearch(page)
    await chooseType(page, 'news')
    await submit(page, 'risk desks behavioural')
    const row = page
      .getByTestId('search-result-news')
      .filter({ hasText: 'Risk desks learn to sit still' })
    await expect(row).toBeVisible({ timeout: 30_000 })
    await expect(row.getByTestId('search-result-news-meta')).toContainText('Daily Example')
    await expect(row.getByTestId('search-result-news-meta')).toContainText('By Ana Example')
    await expect(row.getByTestId('search-result-news-summary')).toContainText('behavioural')
    await expect(row.getByTestId('search-result-news-open')).toHaveAttribute(
      'href',
      /dailyexample\.test\/2026\/11\/risk-desks-learn-to-sit-still/,
    )

    await row.getByTestId('search-result-news-title').click()
    const rail = page.getByTestId('article-detail-rail')
    await expect(rail).toBeVisible()
    await expect(rail.getByTestId('article-title')).toHaveText('Risk desks learn to sit still')
    await expect(rail.getByTestId('article-publisher')).toContainText('dailyexample.test')
    await expect(rail.getByTestId('article-kg-chip').filter({ hasText: 'Risk Management' })).toBeVisible()
    await expect(page.locator('body')).not.toContainText('NEWS-BODY-MARKER')
  })

  test('a teaser is found by its title, with no summary and no body', async ({ page }) => {
    // The soil report is a teaser: news-item-1.0 forbids chunking its body as if whole, so the
    // index holds its title only. That half is not provable in the browser, because the vector
    // leg returns the nearest items for any query; tests/integration/search/test_news_search.py
    // pins it on the index's rows. This test shows the title finds it.
    await openSearch(page)
    await chooseType(page, 'news')
    await submit(page, 'Soil report finds erosion slowing')
    const row = page
      .getByTestId('search-result-news')
      .filter({ hasText: 'Soil report finds erosion slowing' })
    await expect(row).toBeVisible({ timeout: 30_000 })
    await expect(row.getByTestId('search-result-news-summary')).toHaveCount(0)
    await expect(page.locator('body')).not.toContainText('NEWS-BODY-MARKER')
  })
})
