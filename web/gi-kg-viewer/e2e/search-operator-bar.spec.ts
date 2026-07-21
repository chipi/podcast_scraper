import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * Search v3 §S4 (#1234) — ResultSetOperatorBar contract on the Search main tab.
 *
 * Covers S4a (client-only Timeline + On-graph) AND S4b (server-side Cluster +
 * Consensus via /api/search?operator=…). The route handler branches on the
 * ``operator`` query param so one test-suite exercises both the plain top-k
 * path and each operator's dedicated response.
 *
 * The E2E surface map — [E2E_SURFACE_MAP.md](E2E_SURFACE_MAP.md) — is the
 * canonical selector contract; the new testids referenced here are documented
 * in the "Result-set operator bar (#1234)" block of that file.
 */
test.describe('Search — result-set operator bar (#1234)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          search_api: true,
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
    // /api/search branches on the operator query param. Plain search returns
    // the top page; ``operator=cluster`` re-fires with over-fetch (top_k×3)
    // and returns clusters; ``operator=consensus`` returns paired evidence.
    await page.route('**/api/search?**', async (route) => {
      const url = route.request().url()
      const isCluster = /[?&]operator=cluster(?:&|$)/.test(url)
      const isConsensus = /[?&]operator=consensus(?:&|$)/.test(url)

      const baseHits = [
        // kg_topic hit — carries topic_cluster metadata so it lands in a
        // topic_cluster group; also drives the On-graph handoff (source_id).
        {
          doc_id: 'kg_topic:topic:climate',
          score: 0.92,
          source_tier: 'aux',
          text: 'Climate policy',
          metadata: {
            doc_type: 'kg_topic',
            source_id: 'topic:climate',
            topic_label: 'Climate',
            topic_cluster: {
              topic_cluster_compound_id: 'tc:env',
              label: 'Environment',
            },
            publish_date: '2026-04-15',
          },
        },
        // Insight hit — no cluster surface + valid publish month → contributes
        // to the Timeline "2026-04" bucket and falls into the ungrouped bucket.
        {
          doc_id: 'insight:e1:n1',
          score: 0.81,
          source_tier: 'insight',
          text: 'An insight on climate policy',
          metadata: {
            doc_type: 'insight',
            episode_id: 'e1',
            publish_date: '2026-04-30',
          },
        },
        // Transcript hit — no publish_date → increments Timeline "undated" tally.
        {
          doc_id: 'transcript:e2:c1',
          score: 0.7,
          source_tier: 'segment',
          text: 'Raw transcript chunk',
          metadata: { doc_type: 'transcript', episode_id: 'e2' },
        },
      ]

      const body: Record<string, unknown> = {
        query: 'climate',
        query_type: 'semantic',
        results: baseHits,
      }
      if (isCluster) {
        body.operator = 'cluster'
        body.clusters = [
          {
            cluster_id: 'tc:env',
            cluster_kind: 'topic_cluster',
            label: 'Environment',
            size: 1,
            hit_indices: [0],
          },
          {
            cluster_id: null,
            cluster_kind: 'ungrouped',
            label: 'Ungrouped',
            size: 2,
            hit_indices: [1, 2],
          },
        ]
      } else if (isConsensus) {
        body.operator = 'consensus'
        body.consensus_pairs = [
          {
            topic_id: 'topic:climate',
            topic_label: 'Climate',
            person_a_id: 'person:alice',
            person_a_label: 'Alice',
            person_b_id: 'person:bob',
            person_b_label: 'Bob',
            insight_a_id: 'i:a',
            insight_b_id: 'i:b',
            insight_a_text: 'Alice says renewables scale non-linearly.',
            insight_b_text: 'Bob agrees the scaling shape drives costs.',
            contradiction_score: 0.08,
            cosine_similarity: 0.87,
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

  /**
   * Shared entry: land the Search tab, submit a query, wait for the bar. Kept
   * inside the describe block so each test starts from the same UI state.
   */
  async function runSearchAndWaitForBar(page: import('@playwright/test').Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await page.locator('#search-q').fill('climate')
    // Enter submits (SearchPanel's on-keydown handler); avoids the form-linked
    // Search button-scope pattern that broke in an earlier iteration.
    await page.locator('#search-q').press('Enter')
    await expect(page.getByTestId('result-set-operator-bar')).toBeVisible({ timeout: 10_000 })
  }

  test('renders the 4 operator chips (Cluster / Timeline / On graph / Consensus)', async ({
    page,
  }) => {
    await runSearchAndWaitForBar(page)
    await expect(page.getByTestId('operator-chip-cluster')).toBeVisible()
    await expect(page.getByTestId('operator-chip-timeline')).toBeVisible()
    // On-graph chip label includes the resolved id count; 1 kg_topic + 2
    // episode_ids across the 3 mocked hits.
    await expect(page.getByTestId('operator-chip-graph')).toHaveText(/On graph \(3\)/)
    await expect(page.getByTestId('operator-chip-consensus')).toBeVisible()
    await expect(page.getByTestId('operator-chip-cluster')).toBeEnabled()
    await expect(page.getByTestId('operator-chip-consensus')).toBeEnabled()
  })

  test('Timeline: toggles the dot chart on / off; undated tally reflects missing publish_date', async ({
    page,
  }) => {
    await runSearchAndWaitForBar(page)
    await expect(page.getByTestId('operator-timeline-panel')).toHaveCount(0)
    await page.getByTestId('operator-chip-timeline').click()
    const panel = page.getByTestId('operator-timeline-panel')
    await expect(panel).toBeVisible()
    await expect(page.getByTestId('operator-chip-timeline')).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    // One transcript hit has no publish_date → the undated notice renders.
    await expect(page.getByTestId('operator-timeline-undated')).toContainText('1')
    // Second click toggles the panel off; chip returns to unpressed.
    await page.getByTestId('operator-chip-timeline').click()
    await expect(panel).toHaveCount(0)
    await expect(page.getByTestId('operator-chip-timeline')).toHaveAttribute(
      'aria-pressed',
      'false',
    )
  })

  test('On graph: pressing the chip switches to the Graph main tab', async ({ page }) => {
    await runSearchAndWaitForBar(page)
    // Sanity: the Search workspace is visible before the handoff.
    await expect(page.getByTestId('search-workspace')).toBeVisible()
    await page.getByTestId('operator-chip-graph').click()
    // The Search workspace unmounts once ``mainTab === 'graph'`` — that's the
    // load-bearing observable that App.vue's ``activateGraphTab('search')``
    // fired (asserting on ``.graph-canvas`` requires a mocked graph, which is
    // out of scope for the operator-bar contract).
    await expect(page.getByTestId('search-workspace')).toHaveCount(0, { timeout: 10_000 })
    // The Graph tab-panel root is now mounted (empty-state OK when the
    // corpus mock has no graph artifacts).
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
  })

  test('Cluster (server): renders topic-cluster group + ungrouped bucket', async ({ page }) => {
    await runSearchAndWaitForBar(page)
    await page.getByTestId('operator-chip-cluster').click()
    const panel = page.getByTestId('operator-cluster-panel')
    await expect(panel).toBeVisible()
    const rows = page.getByTestId('operator-cluster-list').locator('li')
    await expect(rows).toHaveCount(2)
    // First row = the topic_cluster group (label + size).
    await expect(rows.nth(0)).toContainText('Environment')
    await expect(rows.nth(0)).toContainText('1 hit')
    // Trailing row = ungrouped bucket labelled "Other" per the bar's badge copy.
    await expect(rows.nth(1)).toContainText('Other')
    await expect(rows.nth(1)).toContainText('2 hits')
  })

  test('Cluster: second click on the chip toggles the panel off; no re-fetch', async ({
    page,
  }) => {
    await runSearchAndWaitForBar(page)
    const requests: string[] = []
    page.on('request', (r) => {
      if (r.url().includes('/api/search') && /[?&]operator=cluster/.test(r.url())) {
        requests.push(r.url())
      }
    })
    await page.getByTestId('operator-chip-cluster').click()
    await expect(page.getByTestId('operator-cluster-panel')).toBeVisible()
    await page.getByTestId('operator-chip-cluster').click()
    await expect(page.getByTestId('operator-cluster-panel')).toHaveCount(0)
    // Only the first click fires a request; a toggle-off doesn't re-fetch.
    expect(requests).toHaveLength(1)
  })

  test('Consensus (server): renders pair rows with speaker labels + both scores', async ({
    page,
  }) => {
    await runSearchAndWaitForBar(page)
    await page.getByTestId('operator-chip-consensus').click()
    const panel = page.getByTestId('operator-consensus-panel')
    await expect(panel).toBeVisible()
    const rows = page.getByTestId('operator-consensus-list').locator('li')
    await expect(rows).toHaveCount(1)
    await expect(rows.first()).toContainText('Climate')
    await expect(rows.first()).toContainText('Alice')
    await expect(rows.first()).toContainText('Bob')
    await expect(rows.first()).toContainText('renewables scale non-linearly')
    await expect(rows.first()).toContainText('0.08')
    await expect(rows.first()).toContainText('0.87')
  })

  test('Cluster: over-fetch — the operator request uses top_k × 3', async ({ page }) => {
    await runSearchAndWaitForBar(page)
    const clusterUrl = page.waitForRequest((r) => /[?&]operator=cluster/.test(r.url()))
    await page.getByTestId('operator-chip-cluster').click()
    const req = await clusterUrl
    // Default topK is 10 → operator fires with 30 per RFC-107 §7.4.
    expect(new URL(req.url()).searchParams.get('top_k')).toBe('30')
  })
})
