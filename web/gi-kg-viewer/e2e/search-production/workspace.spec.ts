import { expect, test } from '@playwright/test'
import {
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
  mockSignIn,
} from '../helpers'

/**
 * Search v3 Tier-2 — Query Workspace end-to-end walk (RFC-107, ADR-095
 * §Tier-2). Production-shaped: mocks match what ``GET /api/search``
 * actually returns, not the hand-authored fixture shape in
 * ``e2e/fixtures/production-shaped/search-v3/mocks.json`` (that file
 * predates the S4/S5 implementation — the shipped response wraps
 * ``clusters[]`` / ``consensus_pairs[]`` at the top level and
 * decorates hits with ``metadata.query_enrichments.related_topics``
 * instead of the RFC's originally-planned ``operator_result`` /
 * ``enriched.answer`` fields).
 *
 * **Fixture-refresh follow-up**: regenerate ``mocks.json`` against the
 * shipped shape. Filed as TODO in the fixture directory's README.
 *
 * This spec exercises the entire Search v3 arc in one flow:
 *
 *   S2 (workspace) → S1 (chip filters) → S4b (operator=cluster server)
 *      → S5 (enriched hero from decorated hits) → S4a (Timeline)
 *      → S7 (Recent auto-write) → S3 (palette rehydration)
 *
 * The point of a Tier-2 walk is to catch cross-slice regressions the
 * per-slice specs don't see (e.g. when Recent's ring buffer fights with
 * the palette's live-fetch, or when the operator bar's over-fetch
 * conflicts with the enriched-hero's aggregation).
 */
test.describe('Search v3 Tier-2 — Query Workspace end-to-end walk', () => {
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
    await page.route('**/api/app/preferences', async (route) => {
      const method = route.request().method()
      if (method === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ preferences: {} }),
        })
        return
      }
      if (method === 'PATCH') {
        const req = route.request().postDataJSON() as
          | { preferences?: Record<string, unknown> }
          | null
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ preferences: req?.preferences ?? {} }),
        })
        return
      }
      await route.fulfill({ status: 405, body: '' })
    })
    // Full production-shaped /api/search handler. Branches on operator +
    // enrich_results so a single spec exercises S4b (cluster + consensus)
    // AND S5 (enriched-answer hero decoration). Shipped shapes only:
    //   - ``operator=cluster`` → response.clusters[] + operator="cluster"
    //   - ``operator=consensus`` → response.consensus_pairs[] + operator="consensus"
    //   - ``enrich_results=true`` → per-hit
    //     ``metadata.query_enrichments.related_topics[]``
    //   - Everything else → plain top-k page
    await page.route('**/api/search?**', async (route) => {
      const url = new URL(route.request().url())
      const q = url.searchParams.get('q') ?? ''
      const operator = url.searchParams.get('operator')
      const enrich = url.searchParams.get('enrich_results') === 'true'
      const baseHits = [
        {
          doc_id: 'insight:e1:n1',
          score: 0.92,
          source_tier: 'insight',
          text: 'An insight about compute governance.',
          metadata: {
            doc_type: 'insight',
            episode_id: 'e1',
            episode_title: 'Compute Governance',
            feed_id: 'sha256:mock',
            feed_title: 'Odd Lots (fixture-mock)',
            publish_date: '2026-04-27',
            ...(enrich
              ? {
                  query_enrichments: {
                    related_topics: [
                      { topic_id: 'topic:compute', topic_label: 'Compute', similarity: 0.91 },
                      { topic_id: 'topic:policy', topic_label: 'Policy', similarity: 0.72 },
                    ],
                  },
                }
              : {}),
          },
        },
        {
          doc_id: 'insight:e2:n1',
          score: 0.83,
          source_tier: 'insight',
          text: 'An insight about supply chain risk.',
          metadata: {
            doc_type: 'insight',
            episode_id: 'e2',
            episode_title: 'Supply Chain Risk',
            feed_id: 'sha256:mock',
            feed_title: 'Odd Lots (fixture-mock)',
            publish_date: '2026-05-14',
            ...(enrich
              ? {
                  query_enrichments: {
                    related_topics: [
                      { topic_id: 'topic:supply-chain', topic_label: 'Supply chain', similarity: 0.85 },
                      { topic_id: 'topic:compute', topic_label: 'Compute', similarity: 0.6 },
                    ],
                  },
                }
              : {}),
          },
        },
        {
          doc_id: 'kg_topic:topic:compute',
          score: 0.75,
          source_tier: 'aux',
          text: 'Compute',
          metadata: {
            doc_type: 'kg_topic',
            source_id: 'topic:compute',
            topic_label: 'Compute',
            topic_cluster: {
              topic_cluster_compound_id: 'tc:governance',
              label: 'Governance',
            },
          },
        },
      ]
      const body: Record<string, unknown> = {
        query: q,
        query_type: 'semantic',
        results: baseHits,
      }
      if (operator === 'cluster') {
        body.operator = 'cluster'
        body.clusters = [
          {
            cluster_id: 'tc:governance',
            cluster_kind: 'topic_cluster',
            label: 'Governance',
            size: 1,
            hit_indices: [2],
          },
          {
            cluster_id: null,
            cluster_kind: 'ungrouped',
            label: 'Ungrouped',
            size: 2,
            hit_indices: [0, 1],
          },
        ]
      } else if (operator === 'consensus') {
        body.operator = 'consensus'
        body.consensus_pairs = [
          {
            topic_id: 'topic:compute',
            topic_label: 'Compute',
            person_a_id: 'person:alice',
            person_a_label: 'Alice',
            person_b_id: 'person:bob',
            person_b_label: 'Bob',
            insight_a_id: 'i:a',
            insight_b_id: 'i:b',
            insight_a_text: 'A on compute',
            insight_b_text: 'B agrees on compute',
            contradiction_score: 0.05,
            cosine_similarity: 0.9,
          },
        ]
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(body),
      })
    })
  })

  test('Workspace walk: results → cluster → consensus → timeline → enriched hero → recent write', async ({
    page,
  }) => {
    // ---- S2: Land on the workspace ----
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    const workspace = page.getByTestId('search-workspace')
    await expect(workspace).toBeVisible()

    // ---- S1 + S2: Run the query ----
    await page.locator('#search-q').fill('compute governance')
    await page.locator('#search-q').press('Enter')
    await expect(workspace.locator('article')).toHaveCount(3, { timeout: 10_000 })

    // ---- S4a: Operator bar visible with the 4 chips ----
    await expect(page.getByTestId('result-set-operator-bar')).toBeVisible()
    // On-graph chip resolves to 3 ids (topic + 2 episode fallbacks).
    await expect(page.getByTestId('operator-chip-graph')).toContainText('On graph (3)')

    // ---- S4b: Cluster operator fires + renders ----
    await page.getByTestId('operator-chip-cluster').click()
    const clusterPanel = page.getByTestId('operator-cluster-panel')
    await expect(clusterPanel).toBeVisible()
    const clusterRows = page.getByTestId('operator-cluster-list').locator('li')
    await expect(clusterRows).toHaveCount(2)
    await expect(clusterRows.nth(0)).toContainText('Governance')
    await expect(clusterRows.nth(1)).toContainText('Ungrouped')
    // Toggle off before the next operator.
    await page.getByTestId('operator-chip-cluster').click()
    await expect(clusterPanel).toHaveCount(0)

    // ---- S4b: Consensus operator fires + renders paired evidence ----
    await page.getByTestId('operator-chip-consensus').click()
    const consensusPanel = page.getByTestId('operator-consensus-panel')
    await expect(consensusPanel).toBeVisible()
    const consensusRows = page.getByTestId('operator-consensus-list').locator('li')
    await expect(consensusRows).toHaveCount(1)
    await expect(consensusRows.first()).toContainText('Compute')
    await expect(consensusRows.first()).toContainText('Alice')
    await expect(consensusRows.first()).toContainText('Bob')
    // Close before moving on.
    await page.getByTestId('operator-chip-consensus').click()
    await expect(consensusPanel).toHaveCount(0)

    // ---- S4a: Timeline operator (client-only) ----
    await page.getByTestId('operator-chip-timeline').click()
    const timelinePanel = page.getByTestId('operator-timeline-panel')
    await expect(timelinePanel).toBeVisible()
    await page.getByTestId('operator-chip-timeline').click()
    await expect(timelinePanel).toHaveCount(0)

    // ---- S5: Enriched hero renders topic chips from the enrichment
    // decoration (auto-on since capability + null-filter). ----
    // The initial search already fired enrich_results=true because the
    // chip auto-adopts capability; hits carry query_enrichments.
    // ranked by summed similarity: compute (0.91 + 0.6 = 1.51), supply-chain
    // (0.85), policy (0.72).
    const hero = page.getByTestId('enriched-answer-hero')
    await expect(hero).toBeVisible()
    const heroChips = page.getByTestId('enriched-answer-topics').locator('li')
    await expect(heroChips).toHaveCount(3)
    await expect(heroChips.nth(0)).toContainText('Compute')
    await expect(heroChips.nth(1)).toContainText('Supply chain')
    await expect(heroChips.nth(2)).toContainText('Policy')

    // ---- S7: Recent auto-populated after the search ----
    await expect(page.getByTestId('left-panel-recent-list')).toBeVisible()
    await expect(
      page.getByTestId('left-panel-recent-list').locator('button').first(),
    ).toContainText('compute governance')

    // ---- S3: Palette empty-state reads Recent from USERPREFS-1 ----
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('/')
    await expect(page.getByTestId('command-palette')).toBeVisible()
    await expect(
      page.getByTestId('command-palette-recent-list').locator('button').first(),
    ).toContainText('compute governance')
  })
})
