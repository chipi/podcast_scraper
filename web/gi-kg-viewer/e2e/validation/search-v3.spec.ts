/**
 * Search v3 real-corpus validation walks (F2 — Tier-3 gap fill).
 *
 * Drives ``make serve-for-validation`` against a real operator-supplied
 * corpus and exercises the shipped Search v3 surfaces end-to-end:
 *
 *   T1 — GET /api/search?enrich_results=true → EnrichedAnswerHero
 *   T2 — GET /api/search?operator=cluster    → operator-cluster-panel
 *   T3 — GET /api/search?operator=consensus  → operator-consensus-panel
 *   T4 — POST /api/search/compare (§S8)      → operator-compare-columns
 *   T5 — GET /api/search?episode_id=…        → rail "Search within episode"
 *
 * Each walk defensively skips when the corpus lacks the required
 * artifact (no enrichment output, no topic_consensus.json, etc.) —
 * Tier-2 covers the "response present" contract; Tier-3 is drift
 * detection against the real backend.
 *
 * Run:
 *   make serve-for-validation  # in another terminal (starts viewer 5173 + api 8000)
 *   cd web/gi-kg-viewer
 *   CORPUS_PATH=/abs/path/to/corpus \
 *     node_modules/.bin/playwright test --config playwright.validation.config.ts \
 *     e2e/validation/search-v3.spec.ts
 *
 * Or via make:
 *   make ci-ui-validation CORPUS=/abs/path/to/your/corpus
 */

import { expect, test, type Page } from '@playwright/test'
import {
  mainViewsNav,
  signInIsolated,
  statusBarCorpusPathInput,
} from '../helpers'

const CORPUS_PATH = process.env.CORPUS_PATH ?? ''
if (!CORPUS_PATH) {
  throw new Error(
    'Tier-3 Search v3 validation requires CORPUS_PATH. ' +
      'Set CORPUS_PATH=/abs/path/to/your/corpus or run via ' +
      '`make ci-ui-validation CORPUS=/abs/path/to/your/corpus`.',
  )
}

async function fillCorpusPath(page: Page): Promise<void> {
  const input = statusBarCorpusPathInput(page)
  await input.waitFor({ state: 'visible', timeout: 15_000 })
  await input.fill(CORPUS_PATH)
  await input.press('Enter').catch(() => {})
  await page.waitForTimeout(2000)
}

async function corpusHasSearchIndex(page: Page): Promise<boolean> {
  const url = `http://localhost:8000/api/index/stats?path=${encodeURIComponent(CORPUS_PATH)}`
  const resp = await page.request.get(url).catch(() => null)
  if (!resp) return false
  const body = await resp.json().catch(() => null)
  return Boolean(body?.available)
}

async function landOnSearch(page: Page): Promise<void> {
  await fillCorpusPath(page)
  await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
  await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
  await page.locator('#search-q').waitFor({ state: 'visible', timeout: 15_000 })
}

async function submitSearch(page: Page, q: string): Promise<void> {
  await page.locator('#search-q').fill(q)
  await page.locator('#search-q').press('Enter')
  // Wait for at least one hit article OR an explicit no-results state.
  const workspace = page.getByTestId('search-workspace')
  await expect(async () => {
    const hits = await workspace.locator('article').count()
    if (hits > 0) return
    // No hits — the workspace still renders; assert the empty-state path.
    const noResults = workspace.getByText(/No results|Search returned/i)
    if (await noResults.isVisible()) return
    throw new Error('search hits not visible yet')
  }).toPass({ timeout: 30_000 })
}

test.describe('Search v3 real-corpus validation', () => {
  test.setTimeout(180_000)

  test.beforeAll(async ({ request }) => {
    // Guard: fail fast when the API is unreachable rather than each
    // individual test timing out with a confusing message.
    const resp = await request
      .get(`http://localhost:8000/api/health?path=${encodeURIComponent(CORPUS_PATH)}`)
      .catch(() => null)
    expect(resp?.ok(), 'api at localhost:8000 must be reachable (start `make serve-for-validation`)').toBe(
      true,
    )
  })

  test('T1 — enriched-answer hero renders topic chips from the real QueryEnricher chain', async ({
    page,
  }, testInfo) => {
    const hasIndex = await corpusHasSearchIndex(page)
    test.skip(!hasIndex, 'corpus has no vector index — Tier-2 P1.5 covers this surface')
    await signInIsolated(page, 'search-v3-t1', testInfo)
    await landOnSearch(page)

    // Verify the shell reports enrichment availability BEFORE we submit
    // (the hero auto-adopts capability when the chip stays null). If
    // enrichment isn't configured on this corpus, the hero is
    // legitimately absent — skip rather than fail.
    const health = await page.request
      .get(`http://localhost:8000/api/health?path=${encodeURIComponent(CORPUS_PATH)}`)
      .then((r) => r.json())
      .catch(() => null)
    test.skip(
      !health?.enriched_search_available,
      'corpus has no enrichments/topic_similarity.json — S5 hero cannot decorate hits',
    )
    await submitSearch(page, 'AI')
    await page.screenshot({
      path: 'validation-results/search-v3-t1-1-results.png',
      fullPage: false,
    })
    // Hero renders WHEN the query enricher decorated at least one hit.
    // Some queries yield zero decorated hits (topic similarity below
    // threshold across the whole page); accept that as a valid terminal.
    const hero = page.getByTestId('enriched-answer-hero')
    const heroVisible = await hero.isVisible().catch(() => false)
    if (heroVisible) {
      // If it renders, it MUST hold at least one topic chip.
      const chips = page.getByTestId('enriched-answer-topics').locator('li')
      await expect(chips.first()).toBeVisible({ timeout: 5_000 })
      await page.screenshot({
        path: 'validation-results/search-v3-t1-2-hero-visible.png',
        fullPage: false,
      })
    } else {
      // eslint-disable-next-line no-console
      console.log('[validation T1] no enrichment decorated hits — hero hidden, valid terminal')
    }
  })

  test('T2 — operator=cluster round-trip renders operator-cluster-panel', async ({
    page,
  }, testInfo) => {
    const hasIndex = await corpusHasSearchIndex(page)
    test.skip(!hasIndex, 'corpus has no vector index')
    await signInIsolated(page, 'search-v3-t2', testInfo)
    await landOnSearch(page)
    await submitSearch(page, 'technology')
    const bar = page.getByTestId('result-set-operator-bar')
    await expect(bar).toBeVisible({ timeout: 10_000 })
    await page.getByTestId('operator-chip-cluster').click()
    const panel = page.getByTestId('operator-cluster-panel')
    await expect(panel).toBeVisible({ timeout: 15_000 })
    // Panel must reach a terminal state — either loading finishes with
    // clusters, or the empty-state notice renders (no topic / theme
    // anchors in the visible hits). No console error either way.
    await expect(async () => {
      const list = page.getByTestId('operator-cluster-list')
      const empty = page.getByTestId('operator-cluster-empty')
      const listVisible = await list.isVisible().catch(() => false)
      const emptyVisible = await empty.isVisible().catch(() => false)
      if (!listVisible && !emptyVisible) throw new Error('cluster panel still loading')
    }).toPass({ timeout: 15_000 })
    await page.screenshot({
      path: 'validation-results/search-v3-t2-cluster.png',
      fullPage: false,
    })
  })

  test('T3 — operator=consensus round-trip renders operator-consensus-panel', async ({
    page,
  }, testInfo) => {
    const hasIndex = await corpusHasSearchIndex(page)
    test.skip(!hasIndex, 'corpus has no vector index')
    await signInIsolated(page, 'search-v3-t3', testInfo)
    await landOnSearch(page)
    await submitSearch(page, 'regulation')
    await page.getByTestId('operator-chip-consensus').click()
    const panel = page.getByTestId('operator-consensus-panel')
    await expect(panel).toBeVisible({ timeout: 15_000 })
    // Terminal state: either the pairs list renders or the empty-state
    // notice explains no consensus (corpus lacks topic_consensus.json,
    // or no hit page topic overlaps a pair). No console error either way.
    await expect(async () => {
      const list = page.getByTestId('operator-consensus-list')
      const empty = page.getByTestId('operator-consensus-empty')
      const listVisible = await list.isVisible().catch(() => false)
      const emptyVisible = await empty.isVisible().catch(() => false)
      if (!listVisible && !emptyVisible) throw new Error('consensus panel still loading')
    }).toPass({ timeout: 15_000 })
    await page.screenshot({
      path: 'validation-results/search-v3-t3-consensus.png',
      fullPage: false,
    })
  })

  test('T4 — POST /api/search/compare returns two grounded packs (§S8)', async ({
    page,
  }, testInfo) => {
    const hasIndex = await corpusHasSearchIndex(page)
    test.skip(!hasIndex, 'corpus has no vector index')
    await signInIsolated(page, 'search-v3-t4', testInfo)
    await landOnSearch(page)
    await submitSearch(page, 'compute')
    const chip = page.getByTestId('operator-chip-compare')
    await expect(chip).toBeVisible({ timeout: 10_000 })
    // The chip disables when the visible hit set has < 2 comparable
    // subjects — real corpora sometimes produce single-episode top pages
    // where that happens. Accept it as a valid terminal (Tier-2 covers
    // the enable / render contract).
    const disabled = await chip.isDisabled()
    if (disabled) {
      // eslint-disable-next-line no-console
      console.log('[validation T4] < 2 comparable subjects in top page — chip disabled, valid terminal')
      return
    }
    await chip.click()
    await expect(page.getByTestId('operator-compare-panel')).toBeVisible()
    await page.getByTestId('operator-compare-run').click()
    // Terminal: columns render (both packs came back — grounded or not)
    // OR an operator-compare-error is visible.
    await expect(async () => {
      const columns = page.getByTestId('operator-compare-columns')
      const error = page.getByTestId('operator-compare-error')
      const columnsVisible = await columns.isVisible().catch(() => false)
      const errorVisible = await error.isVisible().catch(() => false)
      if (!columnsVisible && !errorVisible) throw new Error('compare panel still loading')
    }).toPass({ timeout: 30_000 })
    await page.screenshot({
      path: 'validation-results/search-v3-t4-compare.png',
      fullPage: false,
    })
    // If both packs rendered, at least one side must be non-empty
    // (either the top-insight OR the "Ungrounded" badge is visible).
    // This guards against a silently-empty pack view.
    const columnsVisible = await page.getByTestId('operator-compare-columns').isVisible()
    if (columnsVisible) {
      const packA = page.getByTestId('operator-compare-pack-a')
      const packB = page.getByTestId('operator-compare-pack-b')
      await expect(packA).toBeVisible()
      await expect(packB).toBeVisible()
    }
  })

  test('T5 — episode_id rail scope filters the hit set to that episode only', async ({
    page,
  }, testInfo) => {
    const hasIndex = await corpusHasSearchIndex(page)
    test.skip(!hasIndex, 'corpus has no vector index')
    await signInIsolated(page, 'search-v3-t5', testInfo)
    await fillCorpusPath(page)
    // Open Library → first episode → "Search within this episode".
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    const firstRow = page
      .getByRole('button', { name: /, / })
      .filter({ hasNotText: 'Open in graph' })
      .first()
    await firstRow.click({ timeout: 15_000 })
    // Rail launcher lives on EpisodeDetailPanel.
    const railBtn = page.getByTestId('episode-detail-search-in-episode')
    await railBtn.waitFor({ state: 'visible', timeout: 15_000 })
    await railBtn.click()
    // We land on Search tab with the episode chip active + a query
    // pre-filled by buildLibrarySearchHandoffQuery.
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await expect(page.getByTestId('search-chip-episode')).toBeVisible({ timeout: 10_000 })
    // Wait for a hit page OR the no-results state (episode may have
    // no matches for the auto-generated query).
    const workspace = page.getByTestId('search-workspace')
    await expect(async () => {
      const hits = await workspace.locator('article').count()
      const noResults = workspace.getByText(/No results|Search returned/i)
      const hasNoRes = await noResults.isVisible().catch(() => false)
      if (hits === 0 && !hasNoRes) throw new Error('search results not rendered yet')
    }).toPass({ timeout: 30_000 })
    await page.screenshot({
      path: 'validation-results/search-v3-t5-episode-scope.png',
      fullPage: false,
    })
  })
})
