import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * PRD-033 FR1 (#884) — Search/Explore surface: source_tier badges, compound
 * (segment + lifted insight) cards, the Insights/Transcript/Both evidence toggle,
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
 * the three known tiers, each tier the query actually returns must appear, the evidence toggle must
 * narrow to exactly one tier, and Both must restore a superset. Those hold for any healthy result
 * set and still fail if the surface stops rendering tiers, compounds, or entity rows.
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

  test('evidence toggle constrains results to a single tier', async ({ page }) => {
    await runSearch(page)
    const tiers = page.getByTestId('search-result-tier')
    const both = await tiers.count()
    expect(both).toBeGreaterThan(0)

    await page.getByTestId('search-evidence-insight').click()
    await expect.poll(() => tiers.count()).toBeGreaterThan(0)
    for (let i = 0; i < (await tiers.count()); i++) {
      await expect(tiers.nth(i)).toHaveAttribute('data-tier', 'Insight')
    }
    const insightOnly = await tiers.count()

    await page.getByTestId('search-evidence-segment').click()
    await expect.poll(() => tiers.count()).toBeGreaterThan(0)
    for (let i = 0; i < (await tiers.count()); i++) {
      await expect(tiers.nth(i)).toHaveAttribute('data-tier', 'Transcript')
    }
    const segmentOnly = await tiers.count()

    await page.getByTestId('search-evidence-both').click()
    // Both restores a superset of either single tier.
    await expect.poll(() => tiers.count()).toBeGreaterThanOrEqual(
      Math.max(insightOnly, segmentOnly),
    )
  })

})

/**
 * #1619 category B — the corpus cannot produce a **compound** result, so these two keep their stub.
 *
 * A compound row is a `segment` hit whose payload carries a non-null `lifted` block (the insight
 * the chunk was lifted into, plus its speaker/topic). Measured against the live index on
 * 2026-08-14 across five queries — `systems thinking`, `risk management`, `Dr. Elena Fischer`,
 * `lifelong learning`, `expert interviews` — **every** result came back `"lifted": null`, 0 of 50.
 * So the compound badge and the lifted speaker/topic links are unreachable: there is no assertion
 * to rewrite, the surface simply never renders.
 *
 * This is the same gap that blocks `person-landing.spec.ts`, whose only shipped entry point is the
 * lifted speaker link — that file is NOT unblocked by the search index alone. Recorded in
 * docs/wip/CORPUS-V4-FIXTURE-LADDER.md §B.
 */
test.describe('Search FR1 — compound/lifted surfaces (stubbed: corpus produces no lifted blocks)', () => {
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

  test('compound badge marks a segment hit with a lifted insight', async ({ page }) => {
    await runStubbedSearch(page)
    await expect(page.getByTestId('search-result-compound')).toHaveCount(1)
  })

  test('entity names link to a Detail panel', async ({ page }) => {
    await runStubbedSearch(page)
    await expect(page.getByTestId('search-result-lifted-speaker-link')).toBeVisible()
    await expect(page.getByTestId('search-result-lifted-topic-link')).toBeVisible()
    // The standalone "Open Topic panel →" / "Open Person panel →" text link on kg_entity /
    // kg_topic rows was retired (2026-07-22 UX cleanup); the whole row is the affordance now.
    await expect(
      page.getByTestId('search-workspace').locator('article[aria-label="Open Person panel"]'),
    ).toHaveCount(1)
  })
})
