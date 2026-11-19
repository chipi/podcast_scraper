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

/** A podcast topic one shown article and one withheld article also carry (the fixture README). */
const TOPIC = 'topic:risk-management'

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

  test("an article's publisher opens its page in the Library", async ({ page }) => {
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
    await page.getByTestId('article-publisher-open').click()
    await expect(page.getByTestId('library-news-publisher-head')).toContainText('dailyexample.test', {
      timeout: 15_000,
    })
    await expect(page.getByTestId('library-news-list')).not.toContainText('Reliability at scale')
  })

  test('the Library has a News mode with publisher pages', async ({ page }) => {
    await open(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-mode-news').click()
    const list = page.getByTestId('library-news-list')
    await expect(list.getByTestId('search-result-news').first()).toBeVisible({ timeout: 15_000 })
    // News mode never opens a podcast episode in the rail beside a publisher page.
    await expect(page.getByRole('region', { name: 'Episode', exact: true })).toHaveCount(0)
    await expect(list).not.toContainText('WITHHELD-ITEM-TITLE')
    await page.getByTestId('library-news-publisher').selectOption({ label: 'Other Wire' })
    await expect(page.getByTestId('library-news-publisher-head')).toContainText('otherwire.test')
    await expect(list.getByTestId('search-result-news')).toHaveCount(1)
    await expect(list.getByTestId('search-result-news-title')).toHaveText(
      'Reliability at scale, a field report',
    )
    await expect(page.getByRole('region', { name: 'Episode', exact: true })).toHaveCount(0)
    await page.getByTestId('library-mode-podcasts').click()
    await expect(page.getByTestId('library-news-list')).toHaveCount(0)
    // Back to Podcasts with nothing open: the first episode opens, as on a load.
    await expect(page.getByRole('region', { name: 'Episode', exact: true })).toBeVisible()
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

  test("the graph's expand brings the topic's articles as nodes", async ({ page }) => {
    // UXS-019 "The graph": an expand also loads the KG of each shown article that names the node,
    // through /api/artifacts, and draws it. The withheld item's KG is never served.
    await open(page)
    await expect(page.getByTestId('digest-root')).toBeVisible({ timeout: 30_000 })
    await page.evaluate((id) => {
      const subj = (window as unknown as { __GIKG_SUBJECT__?: { focusTopic: (i: string) => void } })
        .__GIKG_SUBJECT__
      if (!subj?.focusTopic) throw new Error('__GIKG_SUBJECT__.focusTopic not exposed (DEV-only)')
      subj.focusTopic(id)
    }, TOPIC)
    await page.getByTestId('node-detail-topic-view-in-graph').click({ timeout: 15_000 })
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
    const cyTopicId = await page.waitForFunction(
      (id) => {
        const cy = (window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }).__GIKG_CY_DEV__
        const hit = cy?.nodes().filter((n) => n.id().replace(/^[gk]:/, '') === id)
        return hit && hit.nonempty() ? hit.first().id() : null
      },
      TOPIC,
      { timeout: 20_000 },
    )
    const articleKgs: { url: string; status: number }[] = []
    page.on('response', (r) => {
      const u = decodeURIComponent(r.url())
      if (u.includes('/api/artifacts/') && u.includes('sources/news_article') && u.includes('.kg.json')) {
        articleKgs.push({ url: u, status: r.status() })
      }
    })
    await page.evaluate((id) => {
      const cy = (window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }).__GIKG_CY_DEV__
      const n = cy!.$id(id)
      n.emit('tap')
      n.emit('dbltap')
    }, String(await cyTopicId.jsonValue()))
    await page.waitForFunction(
      () => {
        const cy = (window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }).__GIKG_CY_DEV__
        return (cy?.nodes().filter((n) => n.id().includes('item:news_article:')).length ?? 0) > 0
      },
      undefined,
      { timeout: 20_000 },
    )
    expect(articleKgs.length).toBeGreaterThan(0)
    expect(articleKgs.every((r) => r.status === 200)).toBe(true)
    expect(articleKgs.some((r) => r.url.includes('withheld'))).toBe(false)
    // The canvas draws no DOM text, so the page check below cannot see a node: ask Cytoscape.
    // Neither the withheld nor the unusable item may be a node (review 2026-11-19).
    const nodeIds = await page.evaluate(() => {
      const cy = (window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }).__GIKG_CY_DEV__
      return cy ? cy.nodes().map((n) => n.id()) : []
    })
    expect(nodeIds.some((id) => id.includes('item:news_article:a8d6e89ce92ef7521ed5'))).toBe(true)
    expect(nodeIds.some((id) => id.includes('item:news_article:2043d8230c86beddfb95'))).toBe(false)
    expect(nodeIds.some((id) => id.includes('item:news_article:772f365b10f0cb5aeae2'))).toBe(false)
    await expect(page.locator('body')).not.toContainText('WITHHELD-ITEM-TITLE')
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
      'News: 4 items from 2 publishers (1 teaser, 1 paywalled)',
      { timeout: 15_000 },
    )
  })
})
