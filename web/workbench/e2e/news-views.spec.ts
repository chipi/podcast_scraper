import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * News in every other view (UXS-019 "Every other view"), live over the app validation corpus's
 * invented news items: the node rail's "In the news", the Library's News mode and publisher page,
 * the Digest's news band and the Dashboard's news counts. No page may show an article body.
 */

async function open(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
}

test.describe('News in every other view (UXS-019, live)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('a topic rail lists the news that names it', async ({ page }) => {
    await open(page)
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await page.getByTestId('search-chip-type').click()
    await page.getByTestId('search-type-option-news').click()
    await page.locator('#search-q').fill('risk desks behavioural')
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()
    await page
      .getByTestId('search-result-news')
      .filter({ hasText: 'Risk desks learn to sit still' })
      .getByTestId('search-result-news-title')
      .click()
    await page.getByTestId('article-kg-chip').filter({ hasText: 'Risk Management' }).click()
    const section = page.getByTestId('node-news-section')
    await expect(section).toBeVisible({ timeout: 15_000 })
    await expect(section.getByTestId('node-news-item').filter({ hasText: 'Risk desks' })).toBeVisible()
    await expect(section).not.toContainText('WITHHELD-ITEM-TITLE')
    await expect(page.locator('body')).not.toContainText('NEWS-BODY-MARKER')
  })

  test('the Library has a News mode with publisher pages', async ({ page }) => {
    await open(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-mode-news').click()
    const list = page.getByTestId('library-news-list')
    await expect(list.getByTestId('search-result-news').first()).toBeVisible({ timeout: 15_000 })
    await expect(list).not.toContainText('WITHHELD-ITEM-TITLE')
    await page.getByTestId('library-news-publisher').selectOption({ label: 'Other Wire' })
    await expect(page.getByTestId('library-news-publisher-head')).toContainText('otherwire.test')
    await expect(list.getByTestId('search-result-news')).toHaveCount(1)
    await expect(list.getByTestId('search-result-news-title')).toHaveText(
      'Reliability at scale, a field report',
    )
    await page.getByTestId('library-mode-podcasts').click()
    await expect(page.getByTestId('library-news-list')).toHaveCount(0)
  })

  test('the Digest has a news band that opens the Library in News mode', async ({ page }) => {
    await open(page)
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    const band = page.getByTestId('digest-news-band')
    await expect(band).toBeVisible({ timeout: 15_000 })
    await expect(band.getByTestId('digest-news-row').first()).toBeVisible()
    await expect(band).not.toContainText('WITHHELD-ITEM-TITLE')
    await band.getByTestId('digest-news-see-all').click()
    await expect(page.getByTestId('library-news-list')).toBeVisible()
  })
})

// The Dashboard tab renders only for an admin (E2E_SURFACE_MAP.md, "Role-gated tabs").
test.describe('News counts on the Dashboard (UXS-019, live, admin)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin', { liveApi: true })
  })

  test('the Dashboard counts the news', async ({ page }) => {
    await open(page)
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await expect(page.getByTestId('dashboard-news-counts')).toHaveText(
      'News: 5 items from 2 publishers (2 teasers, 1 paywalled)',
      { timeout: 15_000 },
    )
  })
})
