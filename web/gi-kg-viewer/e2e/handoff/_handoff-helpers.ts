/**
 * Shared helpers for the graph-handoff matrix specs (`HANDOFF_MATRIX.md`).
 *
 * Standard-assertion contract: every handoff spec asserts the 6-point contract
 * documented in HANDOFF_MATRIX.md §"Standard assertions". This file provides the
 * reusable helpers that implement those assertions against `window.__GIKG_CY_DEV__`
 * and (once C4 lands) `window.__GIKG_STORES__` / `window.__GIKG_FSM__`.
 *
 * Pre-FSM rows (Status: `test.fail()` until C4 / C5 / C7) use only the Cytoscape
 * assertions; FSM-state assertions activate as the FSM is scaffolded.
 */

import { readFileSync } from 'node:fs'
import type { Page } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from '../fixtures'

const ARTIFACT_JSON = readFileSync(GI_SAMPLE_FIXTURE, 'utf-8')

/**
 * Read the live Cytoscape `core` exposed in dev/E2E builds via
 * `window.__GIKG_CY_DEV__`. Returns `null` in production builds where the hook
 * is stripped.
 */
export async function readCyState(page: Page): Promise<{
  totalNodes: number
  selectedIds: string[]
  selectedCount: number
} | null> {
  return page.evaluate(() => {
    const cy = (
      window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
    ).__GIKG_CY_DEV__
    if (!cy) return null
    const selected = cy.nodes(':selected')
    const ids: string[] = []
    selected.forEach((n) => ids.push(n.id()))
    return {
      totalNodes: cy.nodes().length,
      selectedIds: ids,
      selectedCount: selected.length,
    }
  })
}

/**
 * Assert a specific cy id is present in the live core (i.e. `core.$id(id).nonempty()`).
 * Used for validating that a handoff resolved to a node that actually exists in the
 * rendered graph — defends against the `NO_CY_EPISODE_ID` failure mode.
 */
export async function cyIdExists(page: Page, id: string): Promise<boolean> {
  return page.evaluate((targetId) => {
    const cy = (
      window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
    ).__GIKG_CY_DEV__
    if (!cy) return false
    return cy.$id(targetId).nonempty()
  }, id)
}

/**
 * Capture browser console errors emitted during a test. Returns the captured
 * errors (with stack info) when the test ends. Use the standard "no errors"
 * assertion via `expect(errors).toEqual([])`.
 *
 * Caller must invoke once at the start of `test.beforeEach` or at the start of
 * the test body, *before* any user actions.
 */
export function captureConsoleErrors(page: Page): { errors: string[] } {
  const ref = { errors: [] as string[] }
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      ref.errors.push(msg.text())
    }
  })
  return ref
}

/**
 * Set up the corpus / library / digest / episode / search mocks that handoff
 * matrix specs share. Mirrors the patterns from `library.spec.ts`,
 * `digest.spec.ts`, and `search-to-graph-mocks.spec.ts` beforeEach blocks.
 *
 * Provides one episode (`metadata/ep1.metadata.json`, episode_id `e1`).
 *
 * Optional rich fixtures (default off — most matrix specs only need the
 * minimum):
 *   - `digest`: populates `/api/corpus/digest` with one recent row that has
 *     CIL pills targeting ``topic:ci-policy`` (a node already present in
 *     the CI GI fixture), plus a topic band hit referencing the same
 *     episode. Unblocks D1 / D2 / D3 cold-start and hot-state matrix rows.
 *   - `search`: populates `/api/search` with one result wired to the same
 *     ``topic:ci-policy`` node so the search "Show on graph" handoff
 *     reaches the same target. Unblocks S1 rows.
 *   - `clusters`: populates `/api/corpus/topic-clusters` with a cluster
 *     containing ``topic:ci-policy`` so NodeDetail's "Load" / sibling-merge
 *     paths have data to expand. Unblocks O3 / H2.7 / H4.3.
 */
export async function setupHandoffMatrixMocks(
  page: Page,
  opts?: { digest?: boolean; search?: boolean; clusters?: boolean },
): Promise<void> {
  await page.route('**/api/health', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        corpus_library_api: true,
        corpus_digest_api: true,
      }),
    }),
  )
  await page.route('**/api/artifacts?**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
    }),
  )
  await page.route('**/api/corpus/feeds**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        feeds: [{ feed_id: 'f1', display_title: 'Mock Show', episode_count: 1 }],
      }),
    }),
  )
  await page.route('**/api/corpus/episodes**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        feed_id: null,
        items: [
          {
            metadata_relative_path: 'metadata/ep1.metadata.json',
            feed_id: 'f1',
            feed_display_title: 'Mock Show',
            topics: ['Point one'],
            summary_title: 'Summary head',
            summary_bullets_preview: ['Point one'],
            summary_preview: 'Summary head — Point one',
            episode_id: 'ci-fixture',
            episode_title: 'Mock Episode Title',
            publish_date: '2024-06-01',
            gi_relative_path: 'metadata/ep1.gi.json',
            kg_relative_path: 'metadata/ep1.kg.json',
            has_gi: true,
            has_kg: true,
            cil_digest_topics: [],
          },
        ],
        next_cursor: null,
      }),
    }),
  )
  await page.route('**/api/corpus/episodes/detail**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        metadata_relative_path: 'metadata/ep1.metadata.json',
        feed_id: 'f1',
        episode_id: 'ci-fixture',
        episode_title: 'Mock Episode Title',
        publish_date: '2024-06-01',
        summary_title: 'Summary head',
        summary_bullets: ['Point one'],
        summary_text: null,
        gi_relative_path: 'metadata/ep1.gi.json',
        kg_relative_path: 'metadata/ep1.kg.json',
        has_gi: true,
        has_kg: true,
        cil_digest_topics: [],
      }),
    }),
  )
  await page.route('**/api/index/stats**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        available: false,
        reason: 'mock-off',
        stats: null,
        reindex_recommended: false,
      }),
    }),
  )
  await page.route('**/api/corpus/episodes/similar**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        source_metadata_relative_path: 'metadata/ep1.metadata.json',
        query_used: '',
        items: [],
        error: null,
        detail: null,
      }),
    }),
  )
  // Artifact files: serve the CI fixture for both `ep1.gi.json` and
  // `ep1.kg.json` paths so `appendRelativeArtifacts` can complete and the
  // FSM walks through `loading_merge → redrawing_full → applying → ready`.
  // Without these the handoff stalls in `loading_*` and the 5s stuck timer
  // fires (T1 state-walking test would fail).
  await page.route('**/api/artifacts/metadata/ep1.gi.json**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: ARTIFACT_JSON,
    }),
  )
  await page.route('**/api/artifacts/metadata/ep1.kg.json**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: ARTIFACT_JSON,
    }),
  )
  // Topic clusters endpoint
  await page.route('**/api/corpus/topic-clusters**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(
        opts?.clusters
          ? {
              schema_version: '2',
              clusters: [
                {
                  graph_compound_parent_id: 'tc:ci-policy-cluster',
                  cil_alias_target_topic_id: 'topic:ci-policy',
                  canonical_label: 'CI policy cluster',
                  member_count: 1,
                  members: [{ topic_id: 'topic:ci-policy' }],
                },
              ],
              topic_count: 1,
              cluster_count: 1,
              singletons: 0,
            }
          : { path: '/mock/corpus', topic_clusters: [], compounds: [] },
      ),
    }),
  )
  // Digest endpoint
  await page.route('**/api/corpus/digest**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(
        opts?.digest
          ? {
              path: '/mock/corpus',
              window: 'all',
              window_start_utc: '1970-01-01T00:00:00Z',
              window_end_utc: '2024-06-08T00:00:00Z',
              compact: false,
              rows: [
                {
                  metadata_relative_path: 'metadata/ep1.metadata.json',
                  feed_id: 'f1',
                  feed_display_title: 'Mock Show',
                  episode_id: 'ci-fixture',
                  episode_title: 'Mock Episode Title',
                  publish_date: '2024-06-05',
                  summary_title: 'Mock summary',
                  summary_bullets_preview: ['Mock bullet'],
                  summary_bullet_graph_topic_ids: ['topic:ci-policy'],
                  summary_preview: 'Mock summary — Mock bullet',
                  gi_relative_path: 'metadata/ep1.gi.json',
                  kg_relative_path: 'metadata/ep1.kg.json',
                  has_gi: true,
                  has_kg: true,
                  cil_digest_topics: [
                    {
                      topic_id: 'topic:ci-policy',
                      label: 'CI Policy',
                      in_topic_cluster: true,
                      topic_cluster_compound_id: 'tc:ci-policy-cluster',
                    },
                  ],
                },
              ],
              topics: [
                {
                  topic_id: 't1',
                  label: 'CI Policy Band',
                  query: 'ci policy',
                  graph_topic_id: 'topic:ci-policy',
                  hits: [
                    {
                      metadata_relative_path: 'metadata/ep1.metadata.json',
                      episode_title: 'Mock Episode Title',
                      feed_id: 'f1',
                      feed_display_title: 'Mock Show',
                      publish_date: '2024-06-05',
                      score: 0.91,
                      summary_preview: 'Mock summary — Mock bullet',
                      episode_id: 'ci-fixture',
                      gi_relative_path: 'metadata/ep1.gi.json',
                      kg_relative_path: 'metadata/ep1.kg.json',
                      has_gi: true,
                      has_kg: true,
                    },
                  ],
                },
              ],
              topics_unavailable_reason: null,
            }
          : { path: '/mock/corpus', recent: [], topic_bands: [] },
      ),
    }),
  )
  // Search endpoint (only when explicitly enabled)
  if (opts?.search) {
    await page.route('**/api/search?**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: 'ci policy',
          results: [
            {
              doc_id: 'doc-ci-policy',
              score: 0.95,
              text: 'CI policy mention (stub).',
              metadata: {
                doc_type: 'topic',
                source_id: 'topic:ci-policy',
                episode_id: 'ci-fixture',
                source_metadata_relative_path: 'metadata/ep1.metadata.json',
                graph_topic_id: 'topic:ci-policy',
              },
            },
          ],
        }),
      }),
    )
  }
}

/**
 * Reset and read the dev-only FSM event log (T3 contract tests). Returns the
 * captured events with their envelopes; safe to call before user actions to
 * snapshot a clean slate. In production builds the log is undefined.
 */
export async function resetFsmEventLog(page: Page): Promise<void> {
  await page.evaluate(() => {
    const w = window as unknown as { __GIKG_FSM_EVENT_LOG__?: unknown[] }
    w.__GIKG_FSM_EVENT_LOG__ = []
  })
}

/**
 * Read the dev-only FSM event log accumulated since the last `resetFsmEventLog`
 * call. Each entry is the raw `FsmEvent` shape (`{ type, envelope? }` for
 * envelope-carrying events; `{ type }` for parameterless events).
 */
export async function readFsmEventLog(
  page: Page,
): Promise<
  Array<{
    type: string
    envelope?: {
      kind?: string
      source?: string
      loadSource?: string
      cyId?: string
      camera?: { kind?: string }
    }
  }>
> {
  return page.evaluate(() => {
    const w = window as unknown as {
      __GIKG_FSM_EVENT_LOG__?: Array<{ type: string; envelope?: unknown }>
    }
    return (w.__GIKG_FSM_EVENT_LOG__ ?? []).map((e) => ({
      type: e.type,
      envelope: e.envelope as
        | {
            kind?: string
            source?: string
            loadSource?: string
            cyId?: string
            camera?: { kind?: string }
          }
        | undefined,
    }))
  })
}

/**
 * Reset the dev-only FSM state history (T1 state-walking tests). Call before
 * a user action to snapshot a clean slate; afterwards `readFsmStateHistory`
 * returns every state the FSM passed through during the action.
 */
export async function resetFsmStateHistory(page: Page): Promise<void> {
  await page.evaluate(() => {
    const w = window as unknown as { __GIKG_FSM_STATE_HISTORY__?: string[] }
    w.__GIKG_FSM_STATE_HISTORY__ = []
  })
}

/**
 * Read the dev-only FSM state history accumulated since the last
 * `resetFsmStateHistory` call. Captures every state transition (including
 * intermediate `loading_*` / `redrawing_*` / `applying` states that would
 * otherwise be invisible by the time a Playwright test reads
 * `__GIKG_FSM__.state` post-settle).
 */
export async function readFsmStateHistory(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const w = window as unknown as { __GIKG_FSM_STATE_HISTORY__?: string[] }
    return w.__GIKG_FSM_STATE_HISTORY__ ?? []
  })
}

/**
 * Outcome contract for a "successful handoff" — encodes the 6-point standard
 * assertion documented in HANDOFF_MATRIX.md §"Standard assertions". A real
 * end-to-end handoff means more than "an envelope was dispatched"; it means
 * the user actually got the outcome they expected:
 *
 *   1. FSM is back at ``ready`` with ``pending: null`` and
 *      ``lastResult.status === 'applied'``.
 *   2. Cytoscape has exactly one node selected, and its id matches the
 *      expected target (modulo ``g:`` / ``k:`` prefix variants for the
 *      same logical topic / entity).
 *   3. Camera zoom is in a sane range (not collapsed to fit-all).
 *   4. No console errors leaked.
 *   5. For episode handoffs: the Episode panel is visible with the right
 *      title.
 *
 * Use this helper from UI-driven matrix tests so the matrix actually asserts
 * the user-visible outcome, not just the envelope. Dev-hook-driven tests
 * (e.g. Search, NodeDetail, Dashboard — surfaces whose full UI fixture
 * overlaps with other specs) assert against the FSM event log instead;
 * see ``readFsmEventLog``.
 */
export async function assertHandoffApplied(
  page: Page,
  expectedCyId: string,
  opts: {
    errors: { errors: string[] }
    /** Optional: assert the Episode panel shows this title after the handoff. */
    episodePanelTitle?: string
    /** Max time to wait for the FSM to reach ``ready`` (default 15 s). */
    waitForReadyMs?: number
    /** Min zoom to accept (default 0.2 — anything below is "fit-all collapsed"). */
    minZoom?: number
    /** Max zoom to accept (default 5 — anything above is "zoomed into the void"). */
    maxZoom?: number
  },
): Promise<void> {
  const waitMs = opts.waitForReadyMs ?? 15_000
  const minZoom = opts.minZoom ?? 0.2
  const maxZoom = opts.maxZoom ?? 5

  // 1. Wait for FSM to settle.
  const deadline = Date.now() + waitMs
  let fsm: Awaited<ReturnType<typeof readFsmState>> = null
  while (Date.now() < deadline) {
    fsm = await readFsmState(page)
    if (fsm?.state === 'ready' && fsm.pending === null) break
    await page.waitForTimeout(200)
  }
  if (!fsm || fsm.state !== 'ready' || fsm.pending !== null) {
    throw new Error(
      `assertHandoffApplied: FSM did not reach ready within ${waitMs}ms (state=${fsm?.state}, pending=${fsm?.pending?.cyId ?? 'null'})`,
    )
  }
  if (fsm.lastResultStatus !== 'applied') {
    throw new Error(
      `assertHandoffApplied: lastResult.status=${fsm.lastResultStatus} (expected 'applied')`,
    )
  }

  // 2. Selection contract — exactly one node selected, id matches expected.
  const sel = await page.evaluate(() => {
    const cy = (
      window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
    ).__GIKG_CY_DEV__
    if (!cy) return null
    return cy.nodes(':selected').map((n) => n.id())
  })
  if (sel === null) {
    throw new Error('assertHandoffApplied: cy core not exposed via __GIKG_CY_DEV__')
  }
  if (sel.length !== 1) {
    throw new Error(
      `assertHandoffApplied: expected exactly 1 node selected, got ${sel.length} (${JSON.stringify(sel)})`,
    )
  }
  /**
   * Normalize a cy id to its "logical" form so equivalent ids compare
   * equal even if the rendered prefix differs:
   *   - ``g:episode:X``, ``k:episode:X``, ``__unified_ep__:X`` all
   *     normalise to ``episode:X`` (an episode with GI-only, KG-only, or
   *     merged-into-unified data is still the same logical episode)
   *   - ``g:topic:X`` and ``k:topic:X`` normalise to ``topic:X``
   *   - other prefixes (``tc:``, plain ``episode:`` / ``topic:`` / etc.)
   *     pass through unchanged
   */
  const normalize = (s: string): string => {
    if (s.startsWith('__unified_ep__:')) return 'episode:' + s.slice('__unified_ep__:'.length)
    if (s.startsWith('g:') || s.startsWith('k:')) return s.slice(2)
    return s
  }
  if (normalize(sel[0]) !== normalize(expectedCyId)) {
    throw new Error(
      `assertHandoffApplied: selected node ${sel[0]} does not match expected ${expectedCyId} (modulo g:/k:/__unified_ep__: prefix)`,
    )
  }

  // 3. Camera zoom in sane range.
  const zoom = await page.evaluate(() => {
    const cy = (
      window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
    ).__GIKG_CY_DEV__
    return cy ? cy.zoom() : null
  })
  if (zoom === null) {
    throw new Error('assertHandoffApplied: cy core not exposed')
  }
  if (zoom < minZoom || zoom > maxZoom) {
    throw new Error(
      `assertHandoffApplied: zoom=${zoom.toFixed(3)} outside sane range [${minZoom}, ${maxZoom}]`,
    )
  }

  // 4. No console errors.
  if (opts.errors.errors.length > 0) {
    throw new Error(
      `assertHandoffApplied: ${opts.errors.errors.length} console error(s): ${opts.errors.errors.slice(0, 3).join(' | ')}`,
    )
  }

  // 5. Optional: Episode panel visible with right title.
  if (opts.episodePanelTitle) {
    const visible = await page
      .getByRole('region', { name: 'Episode', exact: true })
      .getByRole('heading', { name: opts.episodePanelTitle })
      .isVisible({ timeout: 2000 })
      .catch(() => false)
    if (!visible) {
      throw new Error(
        `assertHandoffApplied: Episode panel not visible with title "${opts.episodePanelTitle}"`,
      )
    }
  }
}

/**
 * Read the live FSM state exposed via `window.__GIKG_FSM__` (dev-only hook
 * stamped by `useGraphHandoffStore`). Returns `null` in production builds where
 * the hook is stripped.
 */
export async function readFsmState(page: Page): Promise<{
  state: string
  pending: { source?: string; kind?: string; cyId?: string; generation?: number } | null
  generation: number
  lastResultStatus: 'applied' | 'failed' | 'superseded' | null
} | null> {
  return page.evaluate(() => {
    const fsm = (
      window as unknown as {
        __GIKG_FSM__?: {
          state: string
          pending: unknown
          generation: number
          lastResult: { status: string; reason?: string } | null
        }
      }
    ).__GIKG_FSM__
    if (!fsm) return null
    const p = fsm.pending as
      | { source?: string; kind?: string; cyId?: string; generation?: number }
      | null
    return {
      state: fsm.state,
      pending: p ?? null,
      generation: fsm.generation,
      lastResultStatus: (fsm.lastResult?.status as
        | 'applied'
        | 'failed'
        | 'superseded'
        | null) ?? null,
    }
  })
}
