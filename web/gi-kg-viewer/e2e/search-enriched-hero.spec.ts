import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * Search v3 §S5 (#1235) — EnrichedAnswerHero contract on the Search main tab.
 *
 * The hero renders an aggregated summary of the shipped QueryEnricher
 * chain (RFC-088 chunk 5) — decorations landed on
 * ``metadata.query_enrichments.related_topics`` per hit. State machine:
 * hidden / skeleton / error / rendered (UXS-008). Bounded-cost surface:
 * NOT rendered when the ``Enriched`` chip is off or the server capability
 * is absent.
 *
 * Owning surface + contract: see the E2E surface map's Search v3 §S5 block.
 */
test.describe('Search — enriched-answer hero (#1235)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          search_api: true,
          // Capability signal that drives the chip's auto-on default.
          enriched_search_available: true,
        }),
      })
    })
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
      })
    })
    // /api/search decorates hits with related_topics only when the caller
    // asks for enrich_results. Plain top-k stays undecorated (proves the
    // hero honors the chip's off state).
    await page.route('**/api/search?**', async (route) => {
      const url = route.request().url()
      const enrich = /[?&]enrich_results=true(?:&|$)/.test(url)
      const results = [
        {
          doc_id: 'insight:e1:n1',
          score: 0.92,
          source_tier: 'insight',
          text: 'A climate insight',
          metadata: {
            doc_type: 'insight',
            episode_id: 'e1',
            publish_date: '2026-04-15',
            ...(enrich
              ? {
                  query_enrichments: {
                    related_topics: [
                      { topic_id: 'topic:climate', topic_label: 'Climate', similarity: 0.9 },
                      { topic_id: 'topic:policy', topic_label: 'Policy', similarity: 0.65 },
                    ],
                  },
                }
              : {}),
          },
        },
        {
          doc_id: 'insight:e2:n1',
          score: 0.81,
          source_tier: 'insight',
          text: 'Another insight',
          metadata: {
            doc_type: 'insight',
            episode_id: 'e2',
            publish_date: '2026-04-30',
            ...(enrich
              ? {
                  query_enrichments: {
                    related_topics: [
                      { topic_id: 'topic:climate', topic_label: 'Climate', similarity: 0.88 },
                    ],
                  },
                }
              : {}),
          },
        },
      ]
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ query: 'climate', query_type: 'semantic', results }),
      })
    })
  })

  async function runSearch(page: import('@playwright/test').Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await page.locator('#search-q').fill('climate')
    await page.locator('#search-q').press('Enter')
    await expect(page.getByTestId('search-workspace').locator('article').first()).toBeVisible()
  }

  test('chip is auto-on when the server advertises enrichment capability', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    const chip = page.getByTestId('search-chip-enriched')
    await expect(chip).toBeVisible()
    await expect(chip).toHaveAttribute('aria-pressed', 'true')
    await expect(chip).toBeEnabled()
  })

  test('hero renders topic chips ranked by summed similarity across hits', async ({ page }) => {
    await runSearch(page)
    const hero = page.getByTestId('enriched-answer-hero')
    await expect(hero).toBeVisible()
    // topic:climate surfaced by 2 hits (summed 1.78); topic:policy by 1.
    const chips = page.getByTestId('enriched-answer-topics').locator('li')
    // 2 topic chips + 0 overflow (2 unique topics ≤ 6 cap).
    await expect(chips).toHaveCount(2)
    await expect(page.getByTestId('enriched-answer-topic-topic:climate')).toBeVisible()
    await expect(page.getByTestId('enriched-answer-topic-topic:policy')).toBeVisible()
    // Climate first (higher summed similarity + hit count).
    const climateFirst = page.getByTestId('enriched-answer-topics').locator('li').first()
    await expect(climateFirst).toContainText('Climate')
    await expect(climateFirst).toContainText('2') // hit count badge
  })

  test('hero is hidden when the Enriched chip is toggled off', async ({ page }) => {
    await runSearch(page)
    await expect(page.getByTestId('enriched-answer-hero')).toBeVisible()
    await page.getByTestId('search-chip-enriched').click()
    // Toggling off does NOT clear the current results, but the hero drops
    // out (renders nothing) because the effective enrichment-on flips.
    await expect(page.getByTestId('enriched-answer-hero')).toHaveCount(0)
  })

  test('hero does NOT render when the server sends no decorated hits (plain top-k)', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    // Turn Enriched OFF so runSearch sends without enrich_results=true.
    await page.getByTestId('search-chip-enriched').click()
    await page.locator('#search-q').fill('climate')
    await page.locator('#search-q').press('Enter')
    await expect(page.getByTestId('search-workspace').locator('article').first()).toBeVisible()
    await expect(page.getByTestId('enriched-answer-hero')).toHaveCount(0)
  })
})
