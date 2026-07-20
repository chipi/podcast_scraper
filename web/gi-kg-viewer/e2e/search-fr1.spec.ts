import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * PRD-033 FR1 (#884) — Search/Explore surface: source_tier badges, compound
 * (segment + lifted insight) cards, the Insights/Transcript/Both evidence toggle,
 * the query-type indicator, and entity names that link to a Detail panel.
 */
test.describe('Search FR1 surfaces (mocked /api/search)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok', corpus_library_api: true }),
      })
    })
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
      })
    })
    await page.route('**/api/search?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: 'Jane Doe climate',
          query_type: 'entity_lookup',
          results: [
            {
              doc_id: 'insight:e1:n1',
              score: 0.9,
              source_tier: 'insight',
              text: 'An insight about climate policy.',
              metadata: { doc_type: 'insight', episode_id: 'e1' },
            },
            {
              doc_id: 'transcript:e1:c2',
              score: 0.85,
              source_tier: 'segment',
              text: 'Raw transcript chunk mentioning climate.',
              metadata: { doc_type: 'transcript', episode_id: 'e1' },
              lifted: {
                insight: { id: 'insight:e1:n1', text: 'Lifted insight text.' },
                speaker: { id: 'person:jane-doe', display_name: 'Jane Doe' },
                topic: { id: 'topic:climate', display_name: 'Climate' },
              },
            },
            {
              doc_id: 'kg_entity:person:jane-doe',
              score: 0.7,
              source_tier: 'aux',
              text: 'Jane Doe',
              metadata: { doc_type: 'kg_entity', source_id: 'person:jane-doe' },
            },
          ],
        }),
      })
    })
  })

  async function runSearch(page: import('@playwright/test').Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await page.locator('#search-q').fill('Jane Doe climate')
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()
    await expect(page.getByTestId('search-result-tier').first()).toBeVisible()
  }

  test('query-type indicator reflects detected intent', async ({ page }) => {
    await runSearch(page)
    await expect(page.getByTestId('search-query-type')).toContainText('Entity lookup')
  })

  test('source_tier badges label each hit', async ({ page }) => {
    await runSearch(page)
    const tiers = page.getByTestId('search-result-tier')
    await expect(tiers.nth(0)).toHaveText('Insight')
    await expect(tiers.nth(1)).toHaveText('Transcript')
    await expect(tiers.nth(2)).toHaveText('Reference')
  })

  test('compound badge marks a segment hit with a lifted insight', async ({ page }) => {
    await runSearch(page)
    await expect(page.getByTestId('search-result-compound')).toHaveCount(1)
  })

  test('evidence toggle constrains results to a tier', async ({ page }) => {
    await runSearch(page)
    await expect(page.getByTestId('search-result-tier')).toHaveCount(3)
    await page.getByTestId('search-evidence-insight').click()
    await expect(page.getByTestId('search-result-tier')).toHaveCount(1)
    await expect(page.getByTestId('search-result-tier').first()).toHaveText('Insight')
    await page.getByTestId('search-evidence-segment').click()
    await expect(page.getByTestId('search-result-tier')).toHaveCount(1)
    await expect(page.getByTestId('search-result-tier').first()).toHaveText('Transcript')
    await page.getByTestId('search-evidence-both').click()
    await expect(page.getByTestId('search-result-tier')).toHaveCount(3)
  })

  test('entity names link to a Detail panel', async ({ page }) => {
    await runSearch(page)
    // Lifted speaker + topic on the compound card, and the kg_entity hit's own link.
    await expect(page.getByTestId('search-result-lifted-speaker-link')).toBeVisible()
    await expect(page.getByTestId('search-result-lifted-topic-link')).toBeVisible()
    await expect(page.getByTestId('search-result-entity-link')).toBeVisible()
  })
})
