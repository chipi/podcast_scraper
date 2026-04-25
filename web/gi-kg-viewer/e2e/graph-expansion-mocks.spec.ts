import { readFileSync } from 'node:fs'
import { expect, test, type Page, type Route } from '@playwright/test'
import { GRAPH_NODE_EPISODES_EXPAND_MAX } from '../src/api/corpusLibraryApi'
import { GI_SAMPLE_FIXTURE } from './fixtures'
import {
  dismissGraphGestureOverlayIfPresent,
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/** Every expand ``POST`` must send the viewer max-episodes cap (``GraphCanvas`` passes this to ``fetchNodeEpisodes``). */
function assertNodeEpisodesExpandPostHasMaxEpisodes(route: Route): void {
  const body = route.request().postDataJSON() as { max_episodes?: number }
  expect(body.max_episodes).toBe(GRAPH_NODE_EPISODES_EXPAND_MAX)
}

/** CI GI fixture plus a second ``ABOUT`` into ``topic:ci-policy`` so Cytoscape ``degree() > 1`` (cross-episode expand gate). */
function giJsonTopicDegreeAtLeastTwo(): string {
  const j = JSON.parse(readFileSync(GI_SAMPLE_FIXTURE, 'utf-8')) as {
    nodes: unknown[]
    edges: unknown[]
  }
  j.nodes.push({
    id: 'insight:gxexp-e2e-degree',
    type: 'Insight',
    properties: {
      text: 'Playwright second edge into topic.',
      episode_id: 'ci-fixture',
      grounded: true,
    },
  })
  j.edges.push(
    {
      type: 'HAS_INSIGHT',
      from: 'episode:ci-fixture',
      to: 'insight:gxexp-e2e-degree',
    },
    {
      type: 'SUPPORTED_BY',
      from: 'insight:gxexp-e2e-degree',
      to: 'quote:4729aa32a95c9ca1',
    },
    {
      type: 'ABOUT',
      from: 'insight:gxexp-e2e-degree',
      to: 'topic:ci-policy',
    },
  )
  return JSON.stringify(j)
}

const secondGiForMerge = JSON.stringify({
  schema_version: '1.0',
  model_version: 'stub',
  prompt_version: 'v1',
  episode_id: 'ep-gxexp-e2e',
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
    {
      id: 'insight:gxexp-e2e-extra',
      type: 'Insight',
      properties: {
        text: 'Extra insight for merged graph size.',
        episode_id: 'ep-gxexp-e2e',
        grounded: true,
      },
    },
    {
      id: 'quote:gxexp-e2e-extra',
      type: 'Quote',
      properties: {
        text: 'Quote body',
        episode_id: 'ep-gxexp-e2e',
        speaker_id: null,
        char_start: 0,
        char_end: 4,
        timestamp_start_ms: 0,
        timestamp_end_ms: 0,
        transcript_ref: 'transcript.txt',
      },
    },
    {
      id: 'topic:ci-policy',
      type: 'Topic',
      properties: { label: 'Climate policy' },
    },
  ],
  edges: [
    {
      type: 'HAS_INSIGHT',
      from: 'episode:ep-gxexp-e2e',
      to: 'insight:gxexp-e2e-extra',
    },
    {
      type: 'SUPPORTED_BY',
      from: 'insight:gxexp-e2e-extra',
      to: 'quote:gxexp-e2e-extra',
    },
    {
      type: 'ABOUT',
      from: 'insight:gxexp-e2e-extra',
      to: 'topic:ci-policy',
    },
  ],
})

const artifactJsonDegree2 = giJsonTopicDegreeAtLeastTwo()

/** Stock CI GI (topic ``degree`` 1) for ineligible expand tests. */
const artifactJsonCiSampleRaw = readFileSync(GI_SAMPLE_FIXTURE, 'utf-8')

function giJsonPersonOrgEligible(): string {
  const j = JSON.parse(artifactJsonDegree2) as { nodes: unknown[]; edges: unknown[] }
  j.nodes.push(
    { id: 'person:gxexp-e2e', type: 'Person', properties: { name: 'Pat' } },
    { id: 'org:gxexp-e2e', type: 'Entity', properties: { name: 'Acme' } },
  )
  j.edges.push(
    { type: 'ABOUT', from: 'insight:b72dafa3f874480d', to: 'person:gxexp-e2e' },
    { type: 'ABOUT', from: 'insight:gxexp-e2e-degree', to: 'person:gxexp-e2e' },
    { type: 'ABOUT', from: 'insight:b72dafa3f874480d', to: 'org:gxexp-e2e' },
    { type: 'ABOUT', from: 'insight:gxexp-e2e-degree', to: 'org:gxexp-e2e' },
  )
  return JSON.stringify(j)
}

const artifactJsonPersonOrg = giJsonPersonOrgEligible()

async function mockGraphExpansionBaseline(page: Page, giArtifactBody: string = artifactJsonDegree2): Promise<void> {
  // Skip first-run graph gesture card so ``.graph-canvas`` clicks are not blocked (Firefox CI).
  await page.addInitScript(() => {
    try {
      window.localStorage?.setItem?.('ps_graph_hints_seen', '1')
    } catch {
      /* ignore private mode / quota */
    }
  })

  await page.route('**/api/health', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        corpus_library_api: true,
        corpus_digest_api: true,
        cil_queries_api: true,
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
            name: 'ci_sample.gi.json',
            relative_path: 'metadata/ci_sample.gi.json',
            kind: 'gi',
            size_bytes: giArtifactBody.length,
            mtime_utc: '2026-04-18T12:00:00Z',
            publish_date: '2026-04-18',
          },
        ],
      }),
    })
  })

  await page.route('**/api/artifacts/metadata/ci_sample.gi.json?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: giArtifactBody,
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

  await page.route('**/api/corpus/topic-clusters**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        schema_version: '2',
        clusters: [],
        topic_count: 0,
        cluster_count: 0,
        singletons: 0,
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
  await page.waitForFunction(() => {
    const el = document.querySelector('.graph-canvas') as HTMLElement | null
    const cy = (window as unknown as { __GIKG_CY_DEV__?: unknown }).__GIKG_CY_DEV__
    return Boolean(cy && el && !el.classList.contains('pointer-events-none'))
  })
  await waitForCyNodeRendered(page, nodeId)
  const pos = await cyRenderedPositionForNodeId(page, nodeId)
  expect(pos).not.toBeNull()
  /** Overlay can appear after ``Fit``; two separate down/up pairs are not a DOM ``dblclick`` (Firefox often never fires Cytoscape ``dbltap``). */
  await dismissGraphGestureOverlayIfPresent(page)
  const canvas = page.locator('.graph-canvas')
  // Two sequential taps (gap) fire ``dbltap`` more reliably than ``clickCount: 2`` on Firefox when
  // ``onetap`` opens the rail — a single compound click can miss the canvas on the second hit.
  await canvas.click({ position: { x: pos!.x, y: pos!.y }, delay: 35 })
  // Wider gap than Cytoscape ``dbltap`` threshold: nightly / Firefox CI can starve rAF under load.
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

async function gotoGraphWithMockCorpus(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill('/mock/corpus')
  await page.getByTestId('status-bar-list-artifacts').waitFor({ state: 'visible', timeout: 15_000 })
  await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
  await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
  await expect(page.locator('.graph-canvas')).toBeVisible()
  await dismissGraphGestureOverlayIfPresent(page)
  await expect
    .poll(async () =>
      page.evaluate(() => {
        const cy = (window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }).__GIKG_CY_DEV__
        if (!cy) return 0
        const n = cy.$id('topic:ci-policy')
        if (n.empty() || typeof n.isNode !== 'function' || !n.isNode()) return 0
        return n.degree()
      }),
    )
    .toBeGreaterThan(1, { timeout: 30_000 })
  // ``scheduleNodeEpisodesCorpusBeyondProbes`` debounces at 400ms; avoid expand racing in-flight POST.
  await page.waitForTimeout(500)
}

test.describe('Graph expansion (mocked API)', () => {
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
              gi_relative_path: 'metadata/gxexp_second.gi.json',
              kg_relative_path: '',
              bridge_relative_path: '',
              episode_id: 'ep-gxexp-e2e',
            },
          ],
          truncated: true,
          total_matched: 12,
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/gxexp_second.gi.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: secondGiForMerge,
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

  test('POST node-episodes receives topic id; expand fetches appended GI', async ({ page }) => {
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
              gi_relative_path: 'metadata/gxexp_second.gi.json',
              kg_relative_path: '',
              bridge_relative_path: '',
              episode_id: 'ep-gxexp-e2e',
            },
          ],
          truncated: false,
          total_matched: null,
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/gxexp_second.gi.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: secondGiForMerge,
      })
    })

    await gotoGraphWithMockCorpus(page)

    const postPromise = page.waitForRequest(
      (r) => r.url().includes('/api/corpus/node-episodes') && r.method() === 'POST',
    )
    const secondGiPromise = page.waitForRequest((r) => r.url().includes('gxexp_second.gi.json'))
    await dblclickCyNode(page, 'topic:ci-policy')
    const postReq = await postPromise
    const body = postReq.postDataJSON() as { node_id?: string; path?: string; max_episodes?: number }
    expect(body.node_id).toBe('topic:ci-policy')
    expect(body.path).toBe('/mock/corpus')
    expect(body.max_episodes).toBe(GRAPH_NODE_EPISODES_EXPAND_MAX)

    const giReq = await secondGiPromise
    expect(giReq.url()).toMatch(/gxexp_second\.gi\.json/i)
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
              gi_relative_path: 'metadata/gxexp_second.gi.json',
              kg_relative_path: '',
              bridge_relative_path: '',
              episode_id: 'ep-gxexp-e2e',
            },
          ],
          truncated: false,
          total_matched: null,
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/gxexp_second.gi.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: secondGiForMerge,
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
    await page.getByRole('checkbox', { name: /Episode/ }).check()
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
})
