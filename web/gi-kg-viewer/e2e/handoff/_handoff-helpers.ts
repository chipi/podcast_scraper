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
    selected.forEach((n) => {
      ids.push(n.id())
    })
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
    // Forward [CI-DIAG] browser logs to test-runner stdout so they appear in
    // CI logs. Temporary — paired with the [CI-DIAG] log emitters in
    // ``GraphCanvas.vue``; remove both when the Tier-2 P1.1/etc. CI-only
    // camera failure is rooted.
    const text = msg.text()
    if (text.startsWith('[CI-DIAG]')) {
      // eslint-disable-next-line no-console
      console.log(text)
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
      body: JSON.stringify({
        path: '/mock/corpus',
        artifacts: [
          {
            name: 'ep1.gi.json',
            relative_path: 'metadata/ep1.gi.json',
            kind: 'gi',
            size_bytes: 1024,
            mtime_utc: '2024-06-01T00:00:00Z',
            publish_date: '2024-06-01',
          },
          {
            name: 'ep1.kg.json',
            relative_path: 'metadata/ep1.kg.json',
            kind: 'kg',
            size_bytes: 1024,
            mtime_utc: '2024-06-01T00:00:00Z',
            publish_date: '2024-06-01',
          },
        ],
      }),
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
                // ``kg_topic`` is in ``FOCUSABLE_DOC_TYPES`` so the result
                // card renders a clickable "Show on graph" (G) button.
                doc_type: 'kg_topic',
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
    /**
     * Skip the camera-centering check (GH #771 class: zoom OK but pan
     * misaligned). Set ``true`` for ``camera: 'fit'`` / ``'preserve'``
     * envelopes where the target node isn't guaranteed to be at the
     * viewport center.
     */
    skipCameraCenter?: boolean
    /**
     * Max distance from viewport center the target node's rendered
     * position may be, as a fraction of viewport dimension. Default 0.35
     * — node may sit anywhere within the inner 70% × 70% of the
     * viewport. Tighten to 0.15 for "really centered" claims; loosen to
     * catch only "off-screen" bugs.
     */
    cameraCenterTolerance?: number
  },
): Promise<void> {
  const waitMs = opts.waitForReadyMs ?? 15_000
  const minZoom = opts.minZoom ?? 0.2
  const maxZoom = opts.maxZoom ?? 5
  const centerTolerance = opts.cameraCenterTolerance ?? 0.35

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

  // 3a. Camera zoom in sane range.
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

  // 3b. Camera-center check (GH #771 class — zoom OK but pan misaligned).
  //
  // ``cy.zoom()`` alone passes for the "fit-all collapsed → zoomed back in
  // but panned to the wrong place" failure mode where the user sees a
  // zoomed view of empty space or wrong nodes. ``node.renderedPosition()``
  // returns the actual pixel coordinates of the node in the viewport after
  // zoom + pan are applied; we require those to be within the inner
  // ``centerTolerance`` × viewport-dimension box around the viewport
  // center.
  //
  // Camera animation runs async after ``recordApplied`` (FSM ``ready`` is
  // marked before ``cy.animate`` finishes), so poll until the position is
  // stable for two consecutive reads, capped at 2.5 s. Skip entirely when
  // the envelope's camera kind isn't centering (``fit`` / ``preserve`` /
  // ``none``) — caller passes ``skipCameraCenter: true``.
  if (!opts.skipCameraCenter) {
    const settled = await page.evaluate(
      async ({ maxMs, pollMs }) => {
        const cy = (
          window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
        ).__GIKG_CY_DEV__
        if (!cy) return null
        const sel = cy.nodes(':selected')
        if (sel.length !== 1) return null
        const node = sel.first()
        let lastX = Number.POSITIVE_INFINITY
        let lastY = Number.POSITIVE_INFINITY
        let stableReads = 0
        const deadline = Date.now() + maxMs
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const rp = node.renderedPosition()
          if (
            Math.abs(rp.x - lastX) < 1 &&
            Math.abs(rp.y - lastY) < 1
          ) {
            stableReads++
            if (stableReads >= 2) {
              return {
                renderedX: rp.x,
                renderedY: rp.y,
                viewportW: cy.width(),
                viewportH: cy.height(),
              }
            }
          } else {
            stableReads = 0
          }
          lastX = rp.x
          lastY = rp.y
          if (Date.now() >= deadline) {
            return {
              renderedX: rp.x,
              renderedY: rp.y,
              viewportW: cy.width(),
              viewportH: cy.height(),
            }
          }
          await new Promise((resolve) => setTimeout(resolve, pollMs))
        }
      },
      { maxMs: 2500, pollMs: 100 },
    )
    if (settled === null) {
      throw new Error(
        'assertHandoffApplied: camera-center check could not read node viewport position',
      )
    }
    const cx = settled.viewportW / 2
    const cy0 = settled.viewportH / 2
    const dx = Math.abs(settled.renderedX - cx)
    const dy = Math.abs(settled.renderedY - cy0)
    const maxDx = settled.viewportW * centerTolerance
    const maxDy = settled.viewportH * centerTolerance
    if (dx > maxDx || dy > maxDy) {
      throw new Error(
        `assertHandoffApplied: target node not centered in viewport. ` +
          `rendered=(${settled.renderedX.toFixed(0)}, ${settled.renderedY.toFixed(0)}) ` +
          `viewportCenter=(${cx.toFixed(0)}, ${cy0.toFixed(0)}) ` +
          `dx=${dx.toFixed(0)}/${maxDx.toFixed(0)} dy=${dy.toFixed(0)}/${maxDy.toFixed(0)} ` +
          `(tolerance=${centerTolerance})`,
      )
    }
  }

  // 4. No console errors.
  if (opts.errors.errors.length > 0) {
    throw new Error(
      `assertHandoffApplied: ${opts.errors.errors.length} console error(s): ${opts.errors.errors.slice(0, 3).join(' | ')}`,
    )
  }

  // 5. Subject-store correctness (L5).
  //
  // The rail panels read the subject store, not cy directly — a "successful
  // handoff" must therefore also leave the subject store reflecting the
  // envelope target. We infer the expected role from the ``expectedCyId``
  // prefix and verify that at least one of the corresponding id fields
  // carries the resolved value.
  //
  // Topic dispatches have two valid shapes today, both internally
  // consistent — keep both legal here, surface the distinction at the
  // FSM-envelope layer (``assertFsmEventEnvelope``) instead:
  //   - ``subject.kind='topic'``  + ``topicId`` set to the bare CIL id
  //     (``topic:ci-policy``). This is what App.activateGraphTab does
  //     when the target arrives WITHOUT a ``g:`` / ``k:`` prefix.
  //   - ``subject.kind='graph-node'`` + ``graphNodeCyId`` set to the full
  //     cy id (``g:topic:ci-policy``). This is what happens when the
  //     target arrives prefixed; the rail still renders the right
  //     surface because NodeDetail dispatches on the resolved cy node.
  const subj = await readSubjectState(page)
  if (subj === null) {
    throw new Error(
      'assertHandoffApplied: subject store not exposed via __GIKG_SUBJECT__',
    )
  }
  const norm = (s: string): string => {
    if (s.startsWith('__unified_ep__:')) return 'episode:' + s.slice('__unified_ep__:'.length)
    if (s.startsWith('g:') || s.startsWith('k:')) return s.slice(2)
    return s
  }
  const normalized = norm(expectedCyId)
  let inferredRole: 'episode' | 'topic' | 'person' | null = null
  if (normalized.startsWith('episode:')) inferredRole = 'episode'
  else if (normalized.startsWith('topic:')) inferredRole = 'topic'
  else if (normalized.startsWith('person:')) inferredRole = 'person'
  // Opaque ids (e.g. ``tc:*`` compounds, bare graph node ids) — don't pin.
  if (inferredRole === 'episode') {
    if (subj.kind !== 'episode') {
      throw new Error(
        `assertHandoffApplied: subject.kind=${subj.kind} (expected episode from cyId ${expectedCyId})`,
      )
    }
    if (!subj.episodeMetadataPath && !subj.episodeId) {
      throw new Error(
        `assertHandoffApplied: subject.kind=episode but both episodeMetadataPath and episodeId are null`,
      )
    }
  } else if (inferredRole === 'topic') {
    // Either canonical topic subject OR graph-node subject pointing at
    // the topic cy id.
    if (subj.kind === 'topic') {
      if (!subj.topicId || norm(subj.topicId) !== normalized) {
        throw new Error(
          `assertHandoffApplied: subject.kind=topic but subject.topicId=${subj.topicId} does not match expected ${expectedCyId}`,
        )
      }
    } else if (subj.kind === 'graph-node') {
      if (!subj.graphNodeCyId || norm(subj.graphNodeCyId) !== normalized) {
        throw new Error(
          `assertHandoffApplied: subject.kind=graph-node but subject.graphNodeCyId=${subj.graphNodeCyId} does not match expected ${expectedCyId}`,
        )
      }
    } else {
      throw new Error(
        `assertHandoffApplied: subject.kind=${subj.kind} (expected topic or graph-node from cyId ${expectedCyId})`,
      )
    }
  } else if (inferredRole === 'person') {
    if (subj.kind !== 'person') {
      throw new Error(
        `assertHandoffApplied: subject.kind=${subj.kind} (expected person from cyId ${expectedCyId})`,
      )
    }
    if (!subj.personId) {
      throw new Error(
        `assertHandoffApplied: subject.kind=person but subject.personId is null`,
      )
    }
  }

  // 6. Self-healing invariant (L6) — set-difference predicate over the
  //    logical view (``viewWithEgo(focusNodeId)``) vs actual Cytoscape
  //    nodes. ``null`` is acceptable: if no layoutstop fired since the
  //    handoff (typical when the target was already in the graph and no
  //    redraw was needed), there's no fresh snapshot to inspect.
  //    Non-null ⇒ both arrays MUST be empty. Production reconciliation
  //    runs the same predicate every layoutstop, so any divergence here
  //    is either an unresolved retry-budget exhaustion or a real bug.
  const inv = await readInvariant(page)
  if (inv !== null) {
    if (inv.missing.length > 0) {
      throw new Error(
        `assertHandoffApplied: invariant violated — ${inv.missing.length} node(s) missing from cy: ${inv.missing.slice(0, 5).join(', ')}`,
      )
    }
    if (inv.extra.length > 0) {
      throw new Error(
        `assertHandoffApplied: invariant violated — ${inv.extra.length} extra node(s) in cy: ${inv.extra.slice(0, 5).join(', ')}`,
      )
    }
  }

  // 7. Optional: Episode panel visible with right title.
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
 * Assert that the FSM event log contains exactly the expected envelope shape
 * for the most recent event of the given type. Used by dev-hook-driven matrix
 * rows (Section 1 entry surfaces with no UI fixture: Search / Dashboard /
 * SubjectRail / StatusBar / NodeDetail / Mini-map) to pin the full FSM event
 * contract — not just ``source``, but also ``kind``, ``loadSource``, and
 * ``camera.kind``.
 *
 * For UI-driven rows, prefer ``assertHandoffApplied`` (the full 6-point
 * outcome contract).
 */
export async function assertFsmEventEnvelope(
  page: Page,
  expect: {
    type: 'handoffRequested' | 'canvasTapped' | 'expansionRequested'
    source: string
    kind?: 'episode' | 'topic' | 'graph-node' | 'person'
    loadSource?: 'subject-external' | 'digest-external' | 'graph-internal'
    cameraKind?: 'center' | 'center-on-target' | 'fit' | 'preserve' | 'none'
    errors: { errors: string[] }
  },
): Promise<void> {
  const log = await readFsmEventLog(page)
  const match = log.find(
    (e) => e.type === expect.type && e.envelope?.source === expect.source,
  )
  if (!match) {
    throw new Error(
      `assertFsmEventEnvelope: no ${expect.type} event with source=${expect.source} in log (${log.length} events: ${log.map((e) => `${e.type}:${e.envelope?.source ?? 'n/a'}`).join(', ')})`,
    )
  }
  const env = match.envelope!
  if (expect.kind !== undefined && env.kind !== expect.kind) {
    throw new Error(
      `assertFsmEventEnvelope: ${expect.type}@${expect.source} kind=${env.kind} (expected ${expect.kind})`,
    )
  }
  if (expect.loadSource !== undefined && env.loadSource !== expect.loadSource) {
    throw new Error(
      `assertFsmEventEnvelope: ${expect.type}@${expect.source} loadSource=${env.loadSource} (expected ${expect.loadSource})`,
    )
  }
  if (expect.cameraKind !== undefined && env.camera?.kind !== expect.cameraKind) {
    throw new Error(
      `assertFsmEventEnvelope: ${expect.type}@${expect.source} camera.kind=${env.camera?.kind} (expected ${expect.cameraKind})`,
    )
  }
  if (expect.errors.errors.length > 0) {
    throw new Error(
      `assertFsmEventEnvelope: ${expect.errors.errors.length} console error(s): ${expect.errors.errors.slice(0, 3).join(' | ')}`,
    )
  }
}

/**
 * Read the live subject-store state exposed via ``window.__GIKG_SUBJECT__``
 * (dev-only hook stamped by ``useSubjectStore``). Returns `null` in production
 * builds where the hook is stripped.
 */
export async function readSubjectState(page: Page): Promise<{
  kind: 'episode' | 'topic' | 'person' | 'graph-node' | null
  episodeMetadataPath: string | null
  episodeId: string | null
  graphNodeCyId: string | null
  topicId: string | null
  personId: string | null
} | null> {
  return page.evaluate(() => {
    const s = (
      window as unknown as {
        __GIKG_SUBJECT__?: {
          kind: 'episode' | 'topic' | 'person' | 'graph-node' | null
          episodeMetadataPath: string | null
          episodeId: string | null
          graphNodeCyId: string | null
          topicId: string | null
          personId: string | null
        }
      }
    ).__GIKG_SUBJECT__
    if (!s) return null
    return {
      kind: s.kind,
      episodeMetadataPath: s.episodeMetadataPath,
      episodeId: s.episodeId,
      graphNodeCyId: s.graphNodeCyId,
      topicId: s.topicId,
      personId: s.personId,
    }
  })
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

/**
 * Read the latest self-healing invariant snapshot stashed by ``finishLayoutPass``
 * (decision #5 / FSM spec section). ``missing`` are nodes the logical view
 * expects but Cytoscape lacks; ``extra`` are nodes Cytoscape has but the
 * logical view doesn't. Both empty ⇒ canvas matches the logical view.
 *
 * Returns ``null`` if no layout pass has run yet OR in production builds.
 */
export async function readInvariant(page: Page): Promise<{
  missing: string[]
  extra: string[]
  ts: number
} | null> {
  return page.evaluate(() => {
    // Prefer sessionStorage — survives Pinia HMR / dev-hook re-stamping mid-test.
    try {
      const raw = window.sessionStorage.getItem('__GIKG_FSM_LAST_INVARIANT__')
      if (raw) {
        return JSON.parse(raw) as { missing: string[]; extra: string[]; ts: number }
      }
    } catch {
      /* fall through to dev-hook getter */
    }
    const fsm = (
      window as unknown as {
        __GIKG_FSM__?: {
          lastInvariant?: { missing: string[]; extra: string[]; ts: number } | null
        }
      }
    ).__GIKG_FSM__
    if (!fsm) return null
    return fsm.lastInvariant ?? null
  })
}
