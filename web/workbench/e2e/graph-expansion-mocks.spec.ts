import { readFileSync } from 'node:fs'
import { expect, test, type Page, type Route } from '@playwright/test'
import { GRAPH_NODE_EPISODES_EXPAND_MAX } from '../src/api/corpusLibraryApi'
import { KG_SAMPLE_FIXTURE } from './fixtures'
import { dismissGraphGestureOverlayIfPresent, mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * #1619 — stays fixture-driven, deliberately. Assessed 2026-08-14 against the live backend.
 *
 * The endpoints are NOT the blocker: `POST /api/corpus/node-episodes` and
 * `POST /api/corpus/resolve-episode-artifacts` both answer for real against the v3 corpus. What
 * this file needs is a graph of a **specific topology** — `kgJsonTopicDegreeAtLeastTwo()` below
 * literally edits `KG_SAMPLE_FIXTURE` to add a second edge into a topic so it reaches
 * `degree() > 1`, because that is the cross-episode expand gate under test. Other tests need a
 * second artifact (`gxexp_second`) to expand into.
 *
 * Those are constructed topologies, not corpus data. Driving them from real corpus artifacts would
 * mean re-deriving every expected expansion from whatever the corpus happens to contain — a large
 * rewrite that makes the assertions weaker and ranking-dependent, in order to test graph mechanics
 * that have nothing to do with the corpus.
 *
 * If revisited: the live endpoints work, so a corpus-driven variant is possible. It would want a
 * fixture corpus containing a known-degree topic, rather than an ad-hoc edit at test time.
 */

/** Every expand ``POST`` must send the viewer max-episodes cap (``GraphCanvas`` passes this to ``fetchNodeEpisodes``). */
function assertNodeEpisodesExpandPostHasMaxEpisodes(route: Route): void {
  const body = route.request().postDataJSON() as { max_episodes?: number }
  expect(body.max_episodes).toBe(GRAPH_NODE_EPISODES_EXPAND_MAX)
}

type KgJson = { nodes: { id: string; type: string; properties?: Record<string, unknown> }[]; edges: unknown[] }

/** CI KG fixture plus a second edge into ``topic:ci-policy`` so Cytoscape ``degree() > 1`` (cross-episode expand gate). */
function kgJsonTopicDegreeAtLeastTwo(): string {
  const j = JSON.parse(readFileSync(KG_SAMPLE_FIXTURE, 'utf-8')) as KgJson
  // From the episode, which is never expand-eligible: an edge from another eligible node would make
  // it probe node-episodes too, and the POST counts below would see it.
  j.edges.push({ type: 'RELATED_TO', from: 'episode:ci-fixture', to: 'topic:ci-policy' })
  return JSON.stringify(j)
}

const secondKgForMerge = JSON.stringify({
  schema_version: '2.0',
  episode_id: 'ep-gxexp-e2e',
  extraction: { extracted_at: '2026-04-17T00:00:00Z', model_version: 'stub', transcript_ref: 'transcript.txt' },
  nodes: [
    {
      id: 'episode:ep-gxexp-e2e',
      type: 'Episode',
      properties: {
        podcast_id: 'podcast:unknown',
        title: 'Graph expansion E2E second episode',
        publish_date: '2026-04-17T00:00:00Z',
      },
    },
    { id: 'person:gxexp-e2e-guest', type: 'Person', properties: { name: 'Guest for merged graph size' } },
    { id: 'topic:ci-policy', type: 'Topic', properties: { label: 'Climate policy' } },
  ],
  edges: [
    { type: 'MENTIONS', from: 'episode:ep-gxexp-e2e', to: 'person:gxexp-e2e-guest' },
    { type: 'MENTIONS', from: 'episode:ep-gxexp-e2e', to: 'topic:ci-policy' },
  ],
})

const artifactJsonDegree2 = kgJsonTopicDegreeAtLeastTwo()

/** Stock CI KG (topic ``degree`` 1) for ineligible expand tests. */
const artifactJsonCiSampleRaw = readFileSync(KG_SAMPLE_FIXTURE, 'utf-8')

function kgJsonPersonOrgEligible(): string {
  const j = JSON.parse(artifactJsonDegree2) as KgJson
  j.nodes.push(
    { id: 'person:gxexp-e2e', type: 'Person', properties: { name: 'Pat' } },
    { id: 'org:gxexp-e2e', type: 'Entity', properties: { name: 'Acme' } },
  )
  // Two edges each, so both clear the expand gate (degree > 1).
  j.edges.push(
    { type: 'MENTIONS', from: 'episode:ci-fixture', to: 'person:gxexp-e2e' },
    { type: 'RELATED_TO', from: 'episode:ci-fixture', to: 'person:gxexp-e2e' },
    { type: 'MENTIONS', from: 'episode:ci-fixture', to: 'org:gxexp-e2e' },
    { type: 'RELATED_TO', from: 'episode:ci-fixture', to: 'org:gxexp-e2e' },
  )
  return JSON.stringify(j)
}

const artifactJsonPersonOrg = kgJsonPersonOrgEligible()

async function mockGraphExpansionBaseline(page: Page, artifactBody: string = artifactJsonDegree2): Promise<void> {
  // Skip first-run graph gesture card so ``.graph-canvas`` clicks are not blocked (Firefox CI).
  await page.addInitScript(() => {
    try {
      window.localStorage?.setItem?.('ps_graph_hints_seen', '1')
    } catch {
      /* ignore private mode / quota */
    }
  })

  await page.route('**/api/health**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        corpus_library_api: true,
        corpus_digest_api: true,
        cil_queries_api: true,
        artifact_kinds: ['kg'],
      }),
    })
  })

  await page.route('**/api/artifacts?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        artifacts: [
          {
            name: 'ci_sample.kg.json',
            relative_path: 'metadata/ci_sample.kg.json',
            kind: 'kg',
            size_bytes: artifactBody.length,
            mtime_utc: '2026-04-18T12:00:00Z',
            publish_date: '2026-04-18',
          },
        ],
      }),
    })
  })

  await page.route('**/api/artifacts/metadata/ci_sample.kg.json?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: artifactBody,
    })
  })

  await page.route('**/api/corpus/feeds**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ path: '/mock/corpus', feeds: [] }),
    })
  })

  await page.route('**/api/index/stats**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        available: false,
        reason: 'mock-off',
        stats: null,
        reindex_recommended: false,
      }),
    })
  })

  await page.route('**/api/corpus/resolve-episode-artifacts**', async (route) => {
    if (route.request().method() !== 'POST') {
      await route.fulfill({ status: 405, body: 'method not allowed' })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        resolved: [],
        missing_episode_ids: [],
      }),
    })
  })

  await page.route('**/api/corpus/stats**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        publish_month_histogram: {},
        catalog_episode_count: 0,
        catalog_feed_count: 0,
        digest_topics_configured: 0,
      }),
    })
  })

  await page.route('**/api/corpus/digest**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        window: 'all',
        window_start_utc: '2020-01-01T00:00:00Z',
        window_end_utc: '2026-01-01T00:00:00Z',
        compact: false,
        rows: [],
        topics: [],
        topics_unavailable_reason: null,
      }),
    })
  })

  // Glob `?**` is a poor match for `…bridge.json?path=…` under Firefox; predicate matches query safely.
  await page.route(
    (url) => url.pathname === '/api/artifacts/metadata/ci_sample.bridge.json',
    async (route) => {
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'no bridge in e2e mock' }),
      })
    },
  )
}

async function cyRenderedPositionForNodeId(
  page: Page,
  nodeId: string,
): Promise<{ x: number; y: number } | null> {
  return page.evaluate((id: string) => {
    const cy = (window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }).__GIKG_CY_DEV__
    if (!cy) {
      return null
    }
    const n = cy.$id(id)
    if (n.empty()) {
      return null
    }
    const p = n.renderedPosition()
    return { x: p.x, y: p.y }
  }, nodeId)
}

async function waitForCyNodeRendered(page: Page, nodeId: string): Promise<void> {
  await expect.poll(async () => cyRenderedPositionForNodeId(page, nodeId)).not.toBeNull()
}

async function dblclickCyNode(page: Page, nodeId: string): Promise<void> {
  // Wait for the canvas + dev cy handle, but don't drive the dbltap via
  // pointer events — Cytoscape's ``dbltap`` detection depends on
  // browser timing that varies wildly between Firefox builds, OS event
  // loops, and CI rAF starvation. The timing-based path was flaky on
  // local Mac (consistent fail) AND nightly Firefox CI (intermittent
  // fail), forcing the gap to be re-tuned every few months.
  //
  // Instead, fire Cytoscape's `tap` then `dbltap` events directly via
  // the dev-mode core handle. This bypasses pointer-event timing
  // entirely and tests exactly the same handler chain (``onetap`` rail
  // open, ``dbltap`` expand) that real users hit.
  //
  // Falls back to the pointer-event path if `__GIKG_CY_DEV__` isn't
  // exposed (production build); both paths land in the same handlers.
  await page.waitForFunction(() => {
    const el = document.querySelector('.graph-canvas') as HTMLElement | null
    const cy = (window as unknown as { __GIKG_CY_DEV__?: unknown }).__GIKG_CY_DEV__
    return Boolean(cy && el && !el.classList.contains('pointer-events-none'))
  })
  await waitForCyNodeRendered(page, nodeId)
  await dismissGraphGestureOverlayIfPresent(page)

  const triggered = await page.evaluate((id: string) => {
    const cy = (window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core })
      .__GIKG_CY_DEV__
    if (!cy) return false
    const n = cy.$id(id)
    if (n.empty()) return false
    // ``onetap`` debounce: fire one tap so the rail opens (matching the
    // real-user single-tap-then-double-tap flow), then the dbltap.
    n.emit('tap')
    n.emit('dbltap')
    return true
  }, nodeId)

  if (triggered) return

  // Fallback for production builds where __GIKG_CY_DEV__ is stripped.
  const pos = await cyRenderedPositionForNodeId(page, nodeId)
  expect(pos).not.toBeNull()
  const canvas = page.locator('.graph-canvas')
  await canvas.click({ position: { x: pos!.x, y: pos!.y }, delay: 35 })
  await page.waitForTimeout(220)
  await canvas.click({ position: { x: pos!.x, y: pos!.y }, delay: 35 })
}

async function clickCyNodeOnce(page: Page, nodeId: string): Promise<void> {
  await page.waitForFunction(() => {
    const el = document.querySelector('.graph-canvas') as HTMLElement | null
    const cy = (window as unknown as { __GIKG_CY_DEV__?: unknown }).__GIKG_CY_DEV__
    return Boolean(cy && el && !el.classList.contains('pointer-events-none'))
  })
  await waitForCyNodeRendered(page, nodeId)
  const pos = await cyRenderedPositionForNodeId(page, nodeId)
  expect(pos).not.toBeNull()
  await dismissGraphGestureOverlayIfPresent(page)
  const canvas = page.locator('.graph-canvas')
  await canvas.click({ position: { x: pos!.x, y: pos!.y }, delay: 30 })
}

async function shiftDblclickCyNode(page: Page, nodeId: string): Promise<void> {
  // Same rationale as `dblclickCyNode` — fire Cytoscape's `dbltap`
  // directly with a synthetic `originalEvent.shiftKey: true` so the
  // ``shift`` branch in the handler runs. Bypasses Firefox modifier-
  // key + double-click timing flake.
  await page.waitForFunction(() => {
    const el = document.querySelector('.graph-canvas') as HTMLElement | null
    const cy = (window as unknown as { __GIKG_CY_DEV__?: unknown }).__GIKG_CY_DEV__
    return Boolean(cy && el && !el.classList.contains('pointer-events-none'))
  })
  await waitForCyNodeRendered(page, nodeId)
  await dismissGraphGestureOverlayIfPresent(page)

  const triggered = await page.evaluate((id: string) => {
    const cy = (window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core })
      .__GIKG_CY_DEV__
    if (!cy) return false
    const n = cy.$id(id)
    if (n.empty()) return false
    // Cytoscape passes `originalEvent` through to handlers; the
    // dbltap site reads `originalEvent.shiftKey`.
    type EmittableEvent = { type: string; originalEvent: { shiftKey: boolean } }
    const evt: EmittableEvent = {
      type: 'dbltap',
      originalEvent: { shiftKey: true },
    }
    // Cast `n.emit` → unknown because cytoscape's typings restrict the
    // payload shape to `EventObject[]` and we want the synthetic event
    // to ride alongside.
    ;(n.emit as unknown as (kind: string, extra: EmittableEvent[]) => void)('dbltap', [evt])
    return true
  }, nodeId)

  if (triggered) return

  // Pointer-event fallback (production builds w/o __GIKG_CY_DEV__).
  const pos = await cyRenderedPositionForNodeId(page, nodeId)
  expect(pos).not.toBeNull()
  const canvas = page.locator('.graph-canvas')
  await canvas.click({
    position: { x: pos!.x, y: pos!.y },
    delay: 35,
    modifiers: ['Shift'],
  })
  await page.waitForTimeout(220)
  await canvas.click({
    position: { x: pos!.x, y: pos!.y },
    delay: 35,
    modifiers: ['Shift'],
  })
}

async function cyNodeCount(page: Page): Promise<number> {
  return page.evaluate(() => {
    const cy = (window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }).__GIKG_CY_DEV__
    return cy ? cy.nodes().length : 0
  })
}

async function gotoGraphWithMockCorpus(
  page: Page,
  opts?: { topicMinDegree?: number },
): Promise<void> {
  // Default: wait for `topic:ci-policy` degree >= 1 (the node exists and
  // renders). Tests that need the cross-episode expand gate open can pass
  // `topicMinDegree: 2` to wait for the second edge from the
  // `kgJsonTopicDegreeAtLeastTwo` fixture; the degree-1 raw-fixture test
  // (which intentionally exercises the *no-expand* path) must NOT block
  // on degree > 1 or it times out.
  const minDegree = opts?.topicMinDegree ?? 1
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill('/mock/corpus')
  await page.getByTestId('status-bar-list-artifacts').waitFor({ state: 'visible', timeout: 15_000 })
  await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
  await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
  await expect(page.locator('.graph-canvas')).toBeVisible()
  await dismissGraphGestureOverlayIfPresent(page)
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const cy = (window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }).__GIKG_CY_DEV__
          if (!cy) return 0
          const n = cy.$id('topic:ci-policy')
          if (n.empty() || typeof n.isNode !== 'function' || !n.isNode()) return 0
          return n.degree()
        }),
      { timeout: 30_000 },
    )
    // Playwright's poll matchers refuse a `{ timeout }` second arg when
    // the expected value is a non-literal expression; pass it via
    // `poll(fn, { timeout })` instead.
    .toBeGreaterThan(minDegree - 1)
  // ``scheduleNodeEpisodesCorpusBeyondProbes`` debounces at 400ms; avoid expand racing in-flight POST.
  await page.waitForTimeout(500)
}

test.describe('Graph expansion (mocked API)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  /** One graph instance per worker; serial avoids overlapping Cytoscape teardown + layout rAF races. */
  test.describe.configure({ mode: 'serial' })
  test('empty node-episodes shows truncation strip; dismiss hides it', async ({ page }) => {
    await mockGraphExpansionBaseline(page)

    await page.route('**/api/corpus/node-episodes**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      assertNodeEpisodesExpandPostHasMaxEpisodes(route)
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          node_id: 'topic:ci-policy',
          episodes: [],
          truncated: false,
          total_matched: null,
        }),
      })
    })

    await gotoGraphWithMockCorpus(page)
    await dblclickCyNode(page, 'topic:ci-policy')

    const strip = page.getByTestId('graph-expansion-truncation-line')
    await expect(strip).toBeVisible({ timeout: 15_000 })
    await expect(strip).toContainText(/No other episodes in the corpus reference this node/i)

    await page.getByTestId('graph-expansion-truncation-dismiss').click()
    await expect(strip).toBeHidden()
  })

  test('truncated node-episodes shows cap copy on Graph tab', async ({ page }) => {
    await mockGraphExpansionBaseline(page)

    await page.route('**/api/corpus/node-episodes**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      assertNodeEpisodesExpandPostHasMaxEpisodes(route)
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          node_id: 'topic:ci-policy',
          episodes: [
            {
              gi_relative_path: '',
              kg_relative_path: 'metadata/gxexp_second.kg.json',
              bridge_relative_path: '',
              episode_id: 'ep-gxexp-e2e',
            },
          ],
          truncated: true,
          total_matched: 12,
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/gxexp_second.kg.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: secondKgForMerge,
      })
    })

    await gotoGraphWithMockCorpus(page)
    const expandDone = page.waitForResponse(
      (res) =>
        res.url().includes('/api/corpus/node-episodes') && res.request().method() === 'POST',
      { timeout: 25_000 },
    )
    await dblclickCyNode(page, 'topic:ci-policy')
    await expandDone

    const strip = page.getByTestId('graph-expansion-truncation-line')
    await expect(strip).toBeVisible({ timeout: 20_000 })
    await expect(strip).toContainText(/Showing 1 of 12 episodes/i)
    await expect(strip).toContainText(/max_episodes cap/i)
  })

  test('POST node-episodes receives topic id; expand fetches appended KG', async ({ page }) => {
    await mockGraphExpansionBaseline(page)

    await page.route('**/api/corpus/node-episodes**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      assertNodeEpisodesExpandPostHasMaxEpisodes(route)
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          node_id: 'topic:ci-policy',
          episodes: [
            {
              gi_relative_path: '',
              kg_relative_path: 'metadata/gxexp_second.kg.json',
              bridge_relative_path: '',
              episode_id: 'ep-gxexp-e2e',
            },
          ],
          truncated: false,
          total_matched: null,
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/gxexp_second.kg.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: secondKgForMerge,
      })
    })

    await gotoGraphWithMockCorpus(page)

    const postPromise = page.waitForRequest(
      (r) => r.url().includes('/api/corpus/node-episodes') && r.method() === 'POST',
    )
    const secondKgPromise = page.waitForRequest((r) => r.url().includes('gxexp_second.kg.json'))
    await dblclickCyNode(page, 'topic:ci-policy')
    const postReq = await postPromise
    const body = postReq.postDataJSON() as { node_id?: string; path?: string; max_episodes?: number }
    expect(body.node_id).toBe('topic:ci-policy')
    expect(body.path).toBe('/mock/corpus')
    expect(body.max_episodes).toBe(GRAPH_NODE_EPISODES_EXPAND_MAX)

    const kgReq = await secondKgPromise
    expect(kgReq.url()).toMatch(/gxexp_second\.kg\.json/i)
  })

  test('node-episodes HTTP error surfaces on truncation strip', async ({ page }) => {
    await mockGraphExpansionBaseline(page)

    await page.route('**/api/corpus/node-episodes**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      assertNodeEpisodesExpandPostHasMaxEpisodes(route)
      await route.fulfill({
        status: 502,
        contentType: 'text/plain; charset=utf-8',
        body: 'node-episodes unavailable (e2e)',
      })
    })

    await gotoGraphWithMockCorpus(page)
    await dblclickCyNode(page, 'topic:ci-policy')

    const strip = page.getByTestId('graph-expansion-truncation-line')
    await expect(strip).toBeVisible({ timeout: 30_000 })
    await expect(strip).toContainText(/node-episodes unavailable \(e2e\)/i)
  })

  test('expand then collapse: only one POST to node-episodes', async ({ page }) => {
    const nodeEpisodesPostCount = { n: 0 }
    page.on('request', (req) => {
      if (req.url().includes('/api/corpus/node-episodes') && req.method() === 'POST') {
        nodeEpisodesPostCount.n += 1
      }
    })
    await mockGraphExpansionBaseline(page)

    await page.route('**/api/corpus/node-episodes**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      assertNodeEpisodesExpandPostHasMaxEpisodes(route)
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          node_id: 'topic:ci-policy',
          episodes: [
            {
              gi_relative_path: '',
              kg_relative_path: 'metadata/gxexp_second.kg.json',
              bridge_relative_path: '',
              episode_id: 'ep-gxexp-e2e',
            },
          ],
          truncated: false,
          total_matched: null,
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/gxexp_second.kg.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: secondKgForMerge,
      })
    })

    await gotoGraphWithMockCorpus(page)
    /** Debounced corpus-beyond probes also POST here; let them finish, then count only expand/collapse. */
    await page.waitForTimeout(600)
    nodeEpisodesPostCount.n = 0

    const firstPost = page.waitForRequest(
      (r) => r.url().includes('/api/corpus/node-episodes') && r.method() === 'POST',
    )
    await dblclickCyNode(page, 'topic:ci-policy')
    await firstPost
    expect(nodeEpisodesPostCount.n).toBe(1)
    await dblclickCyNode(page, 'topic:ci-policy')
    expect(nodeEpisodesPostCount.n).toBe(1)
  })

  test('single activation on topic opens graph node rail (onetap)', async ({ page }) => {
    await mockGraphExpansionBaseline(page)
    await page.route('**/api/corpus/node-episodes**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      assertNodeEpisodesExpandPostHasMaxEpisodes(route)
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          node_id: 'topic:ci-policy',
          episodes: [],
          truncated: false,
          total_matched: null,
        }),
      })
    })

    await gotoGraphWithMockCorpus(page)
    await clickCyNodeOnce(page, 'topic:ci-policy')
    await expect(page.getByTestId('graph-node-detail-rail')).toBeVisible({ timeout: 15_000 })
  })

  test('Shift double-activation on topic toggles ego focus (second shift restores canvas)', async ({
    page,
  }) => {
    await mockGraphExpansionBaseline(page)
    await gotoGraphWithMockCorpus(page)
    const before = await cyNodeCount(page)
    expect(before).toBeGreaterThan(2)
    // Use **Topic** (not **Quote**): quote labels can overlap the hit target in COSE.
    await shiftDblclickCyNode(page, 'topic:ci-policy')
    await shiftDblclickCyNode(page, 'topic:ci-policy')
    await expect.poll(async () => cyNodeCount(page), { timeout: 15_000 }).toBe(before)
  })

  test('Episode node double-activation does not POST node-episodes', async ({ page }) => {
    let postCount = 0
    await mockGraphExpansionBaseline(page)
    await page.route('**/api/corpus/node-episodes**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      postCount += 1
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          node_id: 'episode:ci-fixture',
          episodes: [],
          truncated: false,
          total_matched: null,
        }),
      })
    })
    await gotoGraphWithMockCorpus(page)
    // #658 — Episode type checkbox lives inside the Types chip popover.
    await page.getByTestId('graph-chip-types').click()
    await page.getByTestId('graph-popover-types').getByRole('checkbox', { name: /Episode/ }).check()
    await page.getByTestId('graph-chip-types').click()
    /** Corpus-beyond probes POST independently of the double-click under test. */
    await page.waitForTimeout(600)
    postCount = 0
    await dblclickCyNode(page, 'episode:ci-fixture')
    expect(postCount).toBe(0)
  })

  test('Topic with degree 1 does not POST node-episodes', async ({ page }) => {
    let postCount = 0
    await mockGraphExpansionBaseline(page, artifactJsonCiSampleRaw)
    await page.route('**/api/corpus/node-episodes**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      postCount += 1
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          node_id: 'topic:ci-policy',
          episodes: [],
          truncated: false,
          total_matched: null,
        }),
      })
    })
    await gotoGraphWithMockCorpus(page)
    await page.waitForTimeout(600)
    postCount = 0
    await dblclickCyNode(page, 'topic:ci-policy')
    expect(postCount).toBe(0)
  })

  test('Person and org nodes POST canonical ids to node-episodes', async ({ page }) => {
    const seen: string[] = []
    await mockGraphExpansionBaseline(page, artifactJsonPersonOrg)
    await page.route('**/api/corpus/node-episodes**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      assertNodeEpisodesExpandPostHasMaxEpisodes(route)
      const body = route.request().postDataJSON() as { node_id?: string }
      if (body.node_id) {
        seen.push(body.node_id)
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          node_id: body.node_id ?? '',
          episodes: [],
          truncated: false,
          total_matched: null,
        }),
      })
    })
    await gotoGraphWithMockCorpus(page)
    await dblclickCyNode(page, 'person:gxexp-e2e')
    await expect.poll(() => seen.includes('person:gxexp-e2e')).toBe(true)
    await dblclickCyNode(page, 'org:gxexp-e2e')
    await expect.poll(() => seen.includes('org:gxexp-e2e')).toBe(true)
  })

  test('None on corpus list resets expansion strip', async ({ page }) => {
    await mockGraphExpansionBaseline(page)
    await page.route('**/api/corpus/node-episodes**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      assertNodeEpisodesExpandPostHasMaxEpisodes(route)
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          node_id: 'topic:ci-policy',
          episodes: [],
          truncated: false,
          total_matched: null,
        }),
      })
    })
    await gotoGraphWithMockCorpus(page)
    await dblclickCyNode(page, 'topic:ci-policy')
    const strip = page.getByTestId('graph-expansion-truncation-line')
    await expect(strip).toBeVisible({ timeout: 15_000 })
    await page.getByTestId('status-bar-list-artifacts').click()
    await page.getByTestId('artifact-list-dialog').waitFor({ state: 'visible' })
    await page.getByTestId('artifact-list-dialog').getByRole('button', { name: 'None', exact: true }).click()
    await expect(strip).toBeHidden()
  })

  // #586 regression guard (behaviour-level, runs in browser).
  //
  // Two known triggers for ``TypeError: can't access property "notify",
  // renderer is null`` during the expand flow:
  //
  //  1. ``loadSelected`` cleared ``parsedList`` to ``[]`` at the start
  //     of every load — even on expand (``preserveExpansion: true``) —
  //     causing an intermediate empty-graph ``redraw()`` racing
  //     Cytoscape's pending COSE rAF. **Fixed** in ``artifacts.ts``:
  //     the prior list stays visible until the new one is built; only
  //     a definite empty resolution clears it.
  //
  //  2. Cytoscape's COSE layout fires a post-``layoutstop`` ``fit()``
  //     via rAF that can land after ``cy.destroy()`` if a re-merge
  //     happens during a layout pass. **Not yet fixed** — see follow-up
  //     in #586. The existing GraphCanvas mitigations cap this at
  //     ~2 errors per full expand run, which we accept as the baseline
  //     until the upstream layout race is patched.
  //
  // This test asserts the count does not REGRESS beyond the baseline.
  // Once the second trigger is fixed, tighten the bound to 0 and
  // remove the BASELINE_MAX comment.
  //
  // Pairs with the vitest guard in ``artifacts.loadSelected.test.ts``.
  test('#586: expand does not regress the renderer-null error baseline', async ({ page }) => {
    const consoleErrors: string[] = []
    const pageErrors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text())
    })
    page.on('pageerror', (err) => {
      pageErrors.push(err.message)
    })

    await mockGraphExpansionBaseline(page)
    await page.route('**/api/corpus/node-episodes**', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fulfill({ status: 405, body: 'method not allowed' })
        return
      }
      assertNodeEpisodesExpandPostHasMaxEpisodes(route)
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          node_id: 'topic:ci-policy',
          episodes: [
            {
              gi_relative_path: '',
              kg_relative_path: 'metadata/gxexp_second.kg.json',
              bridge_relative_path: '',
              episode_id: 'ep-gxexp-e2e',
            },
          ],
          truncated: false,
          total_matched: null,
        }),
      })
    })
    await page.route('**/api/artifacts/metadata/gxexp_second.kg.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: secondKgForMerge,
      })
    })

    await gotoGraphWithMockCorpus(page)
    await dblclickCyNode(page, 'topic:ci-policy')

    // Wait for the expanded node to land + let the COSE layout finish a
    // couple of frames past layoutstop so any stray rAF would have
    // fired by now.
    await page.waitForRequest((r) => r.url().includes('gxexp_second.kg.json'))
    await page.waitForTimeout(1500)

    const rendererNullErrors = [...consoleErrors, ...pageErrors].filter((line) =>
      /renderer is null|notify.*renderer|TypeError.*renderer/i.test(line),
    )

    // BASELINE_MAX: ~2 unhandled errors per full expand run, attributed
    // to the COSE post-layoutstop fit() rAF (see test header). Tighten
    // to 0 once that second trigger is fixed. The deferred-parsedList
    // fix (#586 trigger 1) eliminated the empty-graph redraw race; if
    // a regression in that path resurfaces, we'd see this count climb
    // toward 4-6.
    const BASELINE_MAX = 3
    expect(
      rendererNullErrors.length,
      `#586 regression — renderer-null errors exceeded baseline (${BASELINE_MAX}). Got ${rendererNullErrors.length}:\n${rendererNullErrors.join('\n')}`,
    ).toBeLessThanOrEqual(BASELINE_MAX)
  })
})

/**
 * NodeDetail rows the UXS-004 audit found with no browser coverage (2026-10-05). One constructed
 * fixture: the CI KG sample plus topic aliases and a host Person. Opened through the DEV
 * `focusEntity` hook, which is the generic node view. An Insight's supporting-quotes rows are a
 * downstream build's, with GI.
 */
function kgJsonForNodeRailRows(): string {
  const j = JSON.parse(readFileSync(KG_SAMPLE_FIXTURE, 'utf-8')) as KgJson
  const topic = j.nodes.find((n) => n.id === 'topic:ci-policy')!
  topic.properties = { ...(topic.properties ?? {}), aliases: ['climate regulation', ' ', 'carbon policy'] }
  j.nodes.push({ id: 'person:nora', type: 'Person', properties: { name: 'Nora', role: 'host' } })
  return JSON.stringify(j)
}

async function openNodeInRail(page: Page, id: string): Promise<void> {
  await page.evaluate((nodeId) => {
    ;(window as unknown as { __GIKG_SUBJECT__: { focusEntity: (i: string) => void } }).__GIKG_SUBJECT__.focusEntity(nodeId)
  }, id)
  await expect(page.getByTestId('graph-node-detail-rail')).toBeVisible({ timeout: 15_000 })
}

test.describe('Graph node rail rows (UXS-004, fixture-driven)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
    await mockGraphExpansionBaseline(page, kgJsonForNodeRailRows())
    await gotoGraphWithMockCorpus(page)
  })
  test.describe.configure({ mode: 'serial' })

  test('a topic lists its aliases, blanks dropped', async ({ page }) => {
    await openNodeInRail(page, 'topic:ci-policy')
    await expect(page.getByTestId('node-detail-topic-aliases')).toHaveText(
      /Aliases:\s*climate regulation, carbon policy$/,
    )
  })

  test('"Set Search topic filter" opens Search with the topic chip set to this topic', async ({ page }) => {
    await openNodeInRail(page, 'topic:ci-policy')
    await page.getByTestId('node-detail-topic-explore-filter').click()
    await expect(page.getByTestId('search-workspace')).toBeVisible()
    await expect(page.getByTestId('search-chip-topic-contains')).toHaveText('Topic: Climate policy ▾')
  })

  test('a person: the rail heading names the type, and the role reads Host', async ({ page }) => {
    await openNodeInRail(page, 'person:nora')
    // In the rail the type is the panel heading; NodeDetail's own kind row is for the un-embedded
    // view only (`!embedInRail`), which no surface mounts today.
    await expect(page.getByTestId('graph-node-detail-rail').getByRole('heading').first()).toHaveText('Person')
    await expect(page.getByTestId('node-detail-kind-row')).toHaveCount(0)
    await expect(page.getByTestId('node-detail-person-entity-role')).toContainText('Host')
  })

  test('a node outside the loaded graph says its neighbourhood is unavailable', async ({ page }) => {
    await openNodeInRail(page, 'person:not-in-this-graph')
    await page.getByTestId('node-detail-rail-tab-neighbourhood').click()
    await expect(page.getByTestId('node-detail-rail-neighbourhood-unavailable')).toBeVisible()
  })
})
