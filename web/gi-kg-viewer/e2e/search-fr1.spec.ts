import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * PRD-033 FR1 (#884) — Search/Explore surface: source_tier badges,
 * the query-type indicator, and entity names that link to a Detail panel.
 *
 * #1619 — migrated to the live search index.
 *
 * The old version fulfilled a hand-built three-result payload — one per tier, with a lifted block
 * on exactly the middle one — and then asserted those exact counts back. That shape is a fixture,
 * not a finding: it passed whether or not the backend could produce it.
 *
 * Against the real index, ranking decides how many rows come back and in what order, so the
 * assertions here are **relational** rather than absolute: every rendered tier badge must be one of
 * the three known tiers, and each tier the query actually returns must appear. Those hold for any
 * healthy result set and still fail if the surface stops rendering tiers or entity rows. Compound
 * (segment + lifted insight) cards, their speaker/topic links, the Insights/Transcript/Both
 * evidence toggle and the lift line are a downstream build's (E04 step 8b).
 */

/** `source_tier` → the `data-tier` the SearchResultRowIcon carries. */
const TIER_LABELS = ['Insight', 'Transcript', 'Reference'] as const

test.describe('Search FR1 surfaces (live index)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  /** Run a real query through the workspace and wait for rendered results. */
  async function runSearch(page: Page, query = 'systems thinking'): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await page.locator('#search-q').fill(query)
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()
    await expect(page.getByTestId('search-result-tier').first()).toBeVisible({ timeout: 30_000 })
  }

  test('query-type indicator reflects the detected intent', async ({ page }) => {
    await runSearch(page)
    /* The router classifies the query server-side; assert the indicator carries whatever it
     * decided rather than pinning one label, which would be testing the classifier from here. */
    const indicator = page.getByTestId('search-query-type')
    await expect(indicator).toBeVisible()
    await expect(indicator).not.toHaveText('')
  })

  test('source_tier badges label each hit with a known tier', async ({ page }) => {
    await runSearch(page)
    // 2026-07-22 UX cleanup: text tier badges retired in favor of a single leading icon
    // (SearchResultRowIcon). The tier label is still carried as ``data-tier``.
    const tiers = page.getByTestId('search-result-tier')
    const count = await tiers.count()
    expect(count).toBeGreaterThan(0)
    const seen = new Set<string>()
    for (let i = 0; i < count; i++) {
      const tier = await tiers.nth(i).getAttribute('data-tier')
      expect(TIER_LABELS).toContain(tier as (typeof TIER_LABELS)[number])
      if (tier) seen.add(tier)
    }
    /* This corpus answers a broad query across all three tiers — if that stops being true the
     * surface is no longer exercising the tier distinction, and this should say so. */
    expect(seen.size).toBeGreaterThan(1)
  })

  test('the evidence toggle and lift line are not offered (downstream, E04 step 8b)', async ({ page }) => {
    await runSearch(page)
    await expect(page.getByTestId('search-evidence-toggle')).toHaveCount(0)
    await expect(page.getByText(/transcript rows linked to GI/)).toHaveCount(0)
  })

})

/**
 * Stubbed: a segment hit carrying a `lifted` block (the live index returns `"lifted": null`; ladder
 * §B). The compound badge, the lifted block and its speaker/topic links are a downstream build's
 * (E04 step 8b), so here the row renders without them; a kg_entity row still opens its panel.
 */
test.describe('Search FR1 — entity rows; no compound or lifted rows (stubbed)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
    await page.route('**/api/health**', async (route) => {
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

  async function runStubbedSearch(page: Page): Promise<void> {
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

  test('a kg_entity row opens the Person panel; no compound badge or lifted links', async ({ page }) => {
    await runStubbedSearch(page)
    await expect(page.getByTestId('search-result-compound')).toHaveCount(0)
    await expect(page.getByTestId('search-result-lifted-speaker-link')).toHaveCount(0)
    await expect(page.getByTestId('search-result-lifted-topic-link')).toHaveCount(0)
    await expect(page.getByText('Lifted insight text.')).toHaveCount(0)
    // The standalone "Open Topic panel →" / "Open Person panel →" text link on kg_entity /
    // kg_topic rows was retired (2026-07-22 UX cleanup); the whole row is the affordance now.
    await expect(
      page.getByTestId('search-workspace').locator('article[aria-label="Open Person panel"]'),
    ).toHaveCount(1)
  })
})
