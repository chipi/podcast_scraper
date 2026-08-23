import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * Search v3 §S4 (#1234) — ResultSetOperatorBar contract on the Search main tab.
 *
 * Covers S4a (client-only Timeline + On-graph) AND S4b (server-side Cluster + Consensus via
 * ``/api/search?operator=…``).
 *
 * The E2E surface map — [E2E_SURFACE_MAP.md](E2E_SURFACE_MAP.md) — is the canonical selector
 * contract; the testids referenced here are documented in the "Result-set operator bar (#1234)"
 * block of that file.
 *
 * #1619 — migrated to the live index. Both server operators answer for real against the v3
 * corpus: ``operator=cluster`` returns a `theme_cluster` group plus an `ungrouped` bucket, and
 * ``operator=consensus`` returns pair rows. The old version hand-built the response for each
 * branch, so it asserted its own payload back.
 *
 * Where the corpus is thinner than the old fixture, the assertion says so instead of pretending:
 * every live result carries a `publish_date` (so the Timeline "undated" tally is 0, not 1), and
 * consensus pairs come back with `person_a_label` / `person_b_label` / `cosine_similarity` **null**
 * — the speakers are unnamed in this corpus. Those are noted at the tests and recorded in
 * docs/wip/CORPUS-V4-FIXTURE-LADDER.md §B.
 */
const QUERY = 'systems thinking'

test.describe('Search — result-set operator bar (#1234)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  /** Land the Search tab, submit a real query, wait for the operator bar. */
  async function runSearchAndWaitForBar(page: Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await page.locator('#search-q').fill(QUERY)
    // Enter submits (SearchPanel's on-keydown handler); avoids the form-linked
    // Search button-scope pattern that broke in an earlier iteration.
    await page.locator('#search-q').press('Enter')
    await expect(page.getByTestId('result-set-operator-bar')).toBeVisible({ timeout: 30_000 })
  }

  test('renders the 4 operator chips (Cluster / Timeline / On graph / Consensus)', async ({
    page,
  }) => {
    await runSearchAndWaitForBar(page)
    await expect(page.getByTestId('operator-chip-cluster')).toBeVisible()
    await expect(page.getByTestId('operator-chip-timeline')).toBeVisible()
    await expect(page.getByTestId('operator-chip-consensus')).toBeVisible()
    /* The On-graph chip label carries the count of graph-resolvable ids in the result set. That
     * count is ranking-dependent, so assert the shape and that it resolved something — pinning a
     * number would just restate whatever the corpus ranked today. */
    await expect(page.getByTestId('operator-chip-graph')).toHaveText(/On graph \(\d+\)/)
    const graphLabel = await page.getByTestId('operator-chip-graph').textContent()
    expect(Number(/\((\d+)\)/.exec(graphLabel ?? '')?.[1] ?? '0')).toBeGreaterThan(0)
    await expect(page.getByTestId('operator-chip-cluster')).toBeEnabled()
    await expect(page.getByTestId('operator-chip-consensus')).toBeEnabled()
  })

  test('Timeline: toggles the dot chart on / off', async ({ page }) => {
    await runSearchAndWaitForBar(page)
    await expect(page.getByTestId('operator-timeline-panel')).toHaveCount(0)
    await page.getByTestId('operator-chip-timeline').click()
    const panel = page.getByTestId('operator-timeline-panel')
    await expect(panel).toBeVisible()
    await expect(page.getByTestId('operator-chip-timeline')).toHaveAttribute('aria-pressed', 'true')

    /* The old fixture included one hit with no `publish_date` so the "undated" notice rendered.
     * Every result the live corpus returns is dated, so that branch is unreachable here — assert
     * it is absent rather than deleting the coverage silently. A v4 corpus with an undated
     * artifact would flip this.
     *
     * Asserted from the DOM alone, deliberately: confirming it via a second `/api/search` cost an
     * extra query embedding on a single-worker backend for a fact the panel already shows. */
    await expect(page.getByTestId('operator-timeline-undated')).toHaveCount(0)

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
    // The Search workspace unmounts once ``mainTab === 'graph'`` — that's the load-bearing
    // observable that App.vue's ``activateGraphTab('search')`` fired.
    await expect(page.getByTestId('search-workspace')).toHaveCount(0, { timeout: 10_000 })
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
  })

  test('Cluster (server): renders the groups the operator returns, ungrouped last', async ({
    page,
  }) => {
    await runSearchAndWaitForBar(page)

    /* Ask the operator directly with the over-fetch the bar uses, so the expected grouping comes
     * from the server rather than from this file. */
    const resp = await page.request.get(
      `/api/search?q=${encodeURIComponent(QUERY)}&operator=cluster&top_k=30`,
    )
    const { clusters } = (await resp.json()) as {
      clusters: { cluster_kind: string; label: string; size: number }[]
    }
    expect(clusters.length).toBeGreaterThan(1)
    const grouped = clusters.filter((c) => c.cluster_kind !== 'ungrouped')
    expect(grouped.length).toBeGreaterThan(0)

    await page.getByTestId('operator-chip-cluster').click()
    await expect(page.getByTestId('operator-cluster-panel')).toBeVisible()
    const rows = page.getByTestId('operator-cluster-list').locator('li')
    /* Wait for the panel to populate before asserting anything about it: the panel issues its OWN
     * operator request, so it is still empty for a while after the chip click. */
    await expect(rows.first()).toBeVisible({ timeout: 30_000 })
    /* Relational, not equal to `clusters.length`: the probe above and the panel are two SEPARATE
     * clustering runs over a ranked result set, so requiring identical cardinality asserts that
     * two independent computations agreed — which is not the contract and which failed with
     * "expected 2, received 0" while the panel was still loading. What matters is that the panel
     * renders real groups and puts the ungrouped bucket last. */
    await expect(rows).not.toHaveCount(0)
    // The ungrouped bucket renders last, labelled "Other" per the bar's badge copy.
    await expect(rows.last()).toContainText('Other')
    // …and at least one real group is present above it.
    expect(grouped.length).toBeGreaterThan(0)
    await expect(rows.first()).not.toContainText('Other')
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

  test('Consensus (server): renders a pair row per returned pair, with its insight text', async ({
    page,
  }) => {
    await runSearchAndWaitForBar(page)

    const resp = await page.request.get(
      `/api/search?q=${encodeURIComponent(QUERY)}&operator=consensus&top_k=30`,
    )
    const { consensus_pairs: pairs } = (await resp.json()) as {
      consensus_pairs: {
        insight_a_text: string
        person_a_label: string | null
        cosine_similarity: number | null
      }[]
    }
    expect(pairs.length).toBeGreaterThan(0)

    await page.getByTestId('operator-chip-consensus').click()
    await expect(page.getByTestId('operator-consensus-panel')).toBeVisible()
    const rows = page.getByTestId('operator-consensus-list').locator('li')
    /* At least one row, not exactly `pairs.length`: the panel and this probe issue two SEPARATE
     * consensus computations, and pairing is derived from a ranked result set, so the two calls
     * are not guaranteed to agree on cardinality. Equality made this flaky; the contract that
     * matters is that the panel renders the server's pairs at all. */
    await expect(rows.first()).toBeVisible()
    /* Assert on the evidence text, which the corpus does carry. The old fixture also asserted
     * speaker names ("Alice" / "Bob") and a cosine score — this corpus returns
     * `person_a_label`, `person_b_label` and `cosine_similarity` as NULL, so those are unnamed
     * pairs. Ladder §B: v4 should carry a consensus pair with named speakers. */
    await expect(rows.first()).toContainText(pairs[0]!.insight_a_text.slice(0, 40))
    expect(pairs[0]!.person_a_label).toBeNull()
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
