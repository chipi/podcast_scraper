import { expect, test, type Page, type Route } from '@playwright/test'
import { mainViewsNav, mockSignIn, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

/**
 * UXS-008 enriched-answer hero — the states `search-enriched-hero.spec.ts` cannot reach on the
 * live index: loading (skeleton), a non-fatal `enrichment_error`, more than six topics
 * (overflow), the provenance badge, a topic chip opening the topic view, and a server that does
 * not advertise enrichment at all. Each is a response SHAPE, so the search response is mocked.
 */

const QUERY = 'hero states'

type Topic = { topic_id: string; topic_label: string; similarity: number }

function hit(id: string, topics: Topic[]) {
  return {
    doc_id: `insight:${id}`,
    score: 0.8,
    text: `Insight ${id}`,
    metadata: {
      doc_type: 'insight',
      episode_id: `ep-${id}`,
      episode_title: `Episode ${id}`,
      feed_id: 'f1',
      query_enrichments: { related_topics: topics },
    },
  }
}

const EIGHT_TOPICS: Topic[] = Array.from({ length: 8 }, (_, i) => ({
  topic_id: `topic:t${i + 1}`,
  topic_label: `Topic ${i + 1}`,
  similarity: 0.9 - i * 0.05,
}))

async function setup(
  page: Page,
  opts: { capability: boolean; search?: (route: Route) => Promise<void> },
): Promise<void> {
  await mockSignIn(page, 'creator')
  await page.route('**/api/health**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        corpus_library_api: true,
        search_api: true,
        ...(opts.capability ? { enriched_search_available: true } : {}),
      }),
    }),
  )
  if (opts.search) await page.route('**/api/search?**', opts.search)
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill('/mock/corpus')
  await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
  await expect(page.getByTestId('search-workspace')).toBeVisible()
  await page.locator('#search-q').fill(QUERY)
}

function searchBody(results: unknown[], extra: Record<string, unknown> = {}): string {
  return JSON.stringify({ query: QUERY, results, lift_stats: null, error: null, detail: null, ...extra })
}

test.describe('Enriched-answer hero states (UXS-008, mocked)', () => {
  /**
   * Fails on 2026-10-05 — app defect: `SearchPanel.vue` mounts `<EnrichedAnswerHero />` inside
   * `v-if="search.results.length"`, and `stores/search.ts` `runSearch` sets `results = []` before
   * the request. So while a search is in flight the hero is unmounted and its skeleton branch
   * (`isSkeleton`, which needs `search.loading`) can never render — on the first search or any
   * later one.
   */
  test('skeleton while the search is in flight', async ({ page }) => {
    let release: () => void = () => {}
    const gate = new Promise<void>((resolve) => (release = resolve))
    await setup(page, {
      capability: true,
      search: async (route) => {
        await gate
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: searchBody([hit('a', EIGHT_TOPICS.slice(0, 2))]),
        })
      },
    })
    await page.locator('#search-q').press('Enter')
    try {
      await expect(page.getByTestId('enriched-answer-skeleton')).toBeVisible({ timeout: 10_000 })
    } finally {
      release()
    }
    await expect(page.getByTestId('enriched-answer-skeleton')).toHaveCount(0)
    await expect(page.getByTestId('enriched-answer-topics')).toBeVisible()
  })

  test('topics render with the provenance badge, capped at six with an overflow count', async ({
    page,
  }) => {
    await setup(page, {
      capability: true,
      search: (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: searchBody([hit('a', EIGHT_TOPICS.slice(0, 5)), hit('b', EIGHT_TOPICS.slice(3))]),
        }),
    })
    const req = page.waitForRequest((r) => new URL(r.url()).pathname === '/api/search')
    await page.locator('#search-q').press('Enter')
    expect(new URL((await req).url()).searchParams.get('enrich_results')).toBe('true')

    const hero = page.getByTestId('enriched-answer-hero')
    await expect(hero.getByTestId('enriched-answer-provenance')).toHaveText('Deterministic')
    await expect(hero.getByTestId('enriched-answer-topics').locator('li button')).toHaveCount(6)
    await expect(hero.getByTestId('enriched-answer-overflow')).toHaveText('+2 more')
    // Topics 4 and 5 are on both hits, so they lead with a count of 2.
    await expect(hero.getByTestId('enriched-answer-topic-topic:t4')).toContainText('Topic 4')
    await expect(hero.getByTestId('enriched-answer-topic-topic:t4')).toContainText('2')
  })

  test('a topic chip opens that topic in the subject rail', async ({ page }) => {
    await setup(page, {
      capability: true,
      search: (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: searchBody([hit('a', EIGHT_TOPICS.slice(0, 2))]),
        }),
    })
    await page.locator('#search-q').press('Enter')
    await page.getByTestId('enriched-answer-topic-topic:t1').click()
    await expect(page.getByTestId('topic-entity-view')).toBeVisible()
    await expect(page.getByTestId('graph-node-detail-rail').getByRole('heading').first()).toContainText(
      'Topic',
    )
  })

  test('a non-fatal enrichment_error shows the error line, not topics', async ({ page }) => {
    await setup(page, {
      capability: true,
      search: (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: searchBody([hit('a', [])], { enrichment_error: 'enricher timed out' }),
        }),
    })
    await page.locator('#search-q').press('Enter')
    const err = page.getByTestId('enriched-answer-error')
    await expect(err).toBeVisible()
    await expect(err).toHaveText('Enrichment failed for this query — vector hits above are still valid.')
    await expect(page.getByTestId('enriched-answer-topics')).toHaveCount(0)
    await expect(page.getByTestId('search-workspace').locator('article')).toHaveCount(1)
  })

  test('without the server capability the Enriched chip is disabled and says why', async ({
    page,
  }) => {
    await setup(page, {
      capability: false,
      search: (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: searchBody([hit('a', EIGHT_TOPICS.slice(0, 2))]),
        }),
    })
    const chip = page.getByTestId('search-chip-enriched')
    await expect(chip).toBeDisabled()
    await expect(chip).toHaveText('Enriched')
    await expect(chip).toHaveAttribute('title', /^Enrichment not configured on this server/)

    const req = page.waitForRequest((r) => new URL(r.url()).pathname === '/api/search')
    await page.locator('#search-q').press('Enter')
    expect(new URL((await req).url()).searchParams.has('enrich_results')).toBe(false)
    await expect(page.getByTestId('search-workspace').locator('article').first()).toBeVisible()
    await expect(page.getByTestId('enriched-answer-hero')).toHaveCount(0)
  })
})
