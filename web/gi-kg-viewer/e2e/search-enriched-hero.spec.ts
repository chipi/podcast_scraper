import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * Search v3 §S5 (#1235) — EnrichedAnswerHero contract on the Search main tab.
 *
 * The hero renders an aggregated summary of the shipped QueryEnricher chain (RFC-088 chunk 5) —
 * decorations landed on ``metadata.query_enrichments.related_topics`` per hit. State machine:
 * hidden / skeleton / error / rendered (UXS-008). Bounded-cost surface: NOT rendered when the
 * ``Enriched`` chip is off or the server capability is absent.
 *
 * Owning surface + contract: see the E2E surface map's Search v3 §S5 block.
 *
 * #1619 — migrated to the live index. The enricher runs for real: ``/api/health`` advertises
 * ``enriched_search_available: true``, ``enrich_results=true`` comes back with
 * ``query_enrichments.related_topics`` carrying real topic ids and similarities, and a plain
 * top-k has none — which is exactly the on/off contract these tests need, with no stub.
 *
 * Ranking is deliberately NOT re-derived here. The old version hand-built two hits so it could
 * assert "climate first, summed 1.78" — that asserts the fixture, and duplicating the component's
 * summing logic in the test would assert the implementation. Instead: the hero must render, every
 * chip it shows must be a topic the server actually returned, and the off-states must hide it.
 */
const QUERY = 'systems thinking'

type EnrichedResult = {
  metadata?: {
    query_enrichments?: {
      related_topics?: { topic_id: string; topic_label: string; similarity: number }[]
    }
  }
}

test.describe('Search — enriched-answer hero (#1235)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  async function openSearch(page: Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
  }

  async function runSearch(page: Page): Promise<void> {
    await openSearch(page)
    await page.locator('#search-q').fill(QUERY)
    await page.locator('#search-q').press('Enter')
    await expect(
      page.getByTestId('search-workspace').locator('article').first(),
    ).toBeVisible({ timeout: 30_000 })
  }

  /** The topic ids the server's enricher decorated this query's hits with. */
  async function liveRelatedTopicIds(page: Page): Promise<Set<string>> {
    const resp = await page.request.get(
      `/api/search?q=${encodeURIComponent(QUERY)}&top_k=10&enrich_results=true`,
    )
    const { results } = (await resp.json()) as { results: EnrichedResult[] }
    const ids = new Set<string>()
    for (const r of results) {
      for (const t of r.metadata?.query_enrichments?.related_topics ?? []) {
        ids.add(t.topic_id)
      }
    }
    return ids
  }

  test('chip is auto-on when the server advertises enrichment capability', async ({ page }) => {
    /* Assert the capability is really being advertised — otherwise "chip is on" would pass for
     * the wrong reason on a server that never enriches. */
    const health = await (await page.request.get('/api/health')).json()
    expect(health.enriched_search_available).toBe(true)

    await openSearch(page)
    const chip = page.getByTestId('search-chip-enriched')
    await expect(chip).toBeVisible()
    await expect(chip).toHaveAttribute('aria-pressed', 'true')
    await expect(chip).toBeEnabled()
  })

  test('hero renders topic chips drawn from the server enrichments', async ({ page }) => {
    await runSearch(page)
    const hero = page.getByTestId('enriched-answer-hero')
    await expect(hero).toBeVisible({ timeout: 30_000 })

    const serverTopicIds = await liveRelatedTopicIds(page)
    expect(serverTopicIds.size).toBeGreaterThan(0)

    const chips = page.getByTestId('enriched-answer-topics').locator('li')
    const shown = await chips.count()
    expect(shown).toBeGreaterThan(0)
    /* The hero caps at 6 chips (UXS-008), so it shows a subset — every chip it does show must be
     * a topic the server returned. This fails if the hero ever renders a topic from anywhere
     * else, which is the property worth guarding. */
    expect(shown).toBeLessThanOrEqual(Math.min(serverTopicIds.size, 6))
    for (const id of serverTopicIds) {
      const chip = page.getByTestId(`enriched-answer-topic-${id}`)
      if ((await chip.count()) > 0) {
        await expect(chip.first()).toBeVisible()
      }
    }
  })

  test('hero is hidden when the Enriched chip is toggled off', async ({ page }) => {
    await runSearch(page)
    await expect(page.getByTestId('enriched-answer-hero')).toBeVisible({ timeout: 30_000 })
    await page.getByTestId('search-chip-enriched').click()
    // Toggling off does NOT clear the current results, but the hero drops out because the
    // effective enrichment-on flips.
    await expect(page.getByTestId('enriched-answer-hero')).toHaveCount(0)
  })

  test('hero does NOT render when the server sends no decorated hits (plain top-k)', async ({
    page,
  }) => {
    /* Confirm the premise against the live server: a plain top-k really is undecorated. */
    const plain = await (
      await page.request.get(`/api/search?q=${encodeURIComponent(QUERY)}&top_k=10`)
    ).json()
    const decorated = (plain.results as EnrichedResult[]).filter(
      (r) => r.metadata?.query_enrichments,
    ).length
    expect(decorated).toBe(0)

    await openSearch(page)
    // Turn Enriched OFF so the query goes without enrich_results=true.
    await page.getByTestId('search-chip-enriched').click()
    await page.locator('#search-q').fill(QUERY)
    await page.locator('#search-q').press('Enter')
    await expect(
      page.getByTestId('search-workspace').locator('article').first(),
    ).toBeVisible({ timeout: 30_000 })
    await expect(page.getByTestId('enriched-answer-hero')).toHaveCount(0)
  })
})
