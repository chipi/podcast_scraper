import { readFileSync } from 'node:fs'
import { expect, test, type Page, type Route } from '@playwright/test'
import { RFC076_EXPAND_MAX_EPISODES } from '../src/api/corpusLibraryApi'
import { GI_SAMPLE_FIXTURE } from './fixtures'
import { mainViewsNav, SHELL_HEADING_RE } from './helpers'

/** Every expand ``POST`` must send the RFC-076 viewer cap (``GraphCanvas`` passes this to ``fetchNodeEpisodes``). */
function assertNodeEpisodesExpandPostHasMaxEpisodes(route: Route): void {
  const body = route.request().postDataJSON() as { max_episodes?: number }
  expect(body.max_episodes).toBe(RFC076_EXPAND_MAX_EPISODES)
}

/** CI GI fixture plus a second ``ABOUT`` into ``topic:ci-policy`` so Cytoscape ``degree() > 1`` (RFC-076 expand gate). */
function giJsonTopicDegreeAtLeastTwo(): string {
  const j = JSON.parse(readFileSync(GI_SAMPLE_FIXTURE, 'utf-8')) as {
    nodes: unknown[]
    edges: unknown[]
  }
  j.nodes.push({
    id: 'insight:rfc076-e2e-degree',
    type: 'Insight',
    properties: {
      text: 'Playwright RFC-076 second edge into topic.',
      episode_id: 'ci-fixture',
      grounded: true,
    },
  })
  j.edges.push(
    {
      type: 'HAS_INSIGHT',
      from: 'episode:ci-fixture',
      to: 'insight:rfc076-e2e-degree',
    },
    {
      type: 'SUPPORTED_BY',
      from: 'insight:rfc076-e2e-degree',
      to: 'quote:4729aa32a95c9ca1',
    },
    {
      type: 'ABOUT',
      from: 'insight:rfc076-e2e-degree',
      to: 'topic:ci-policy',
    },
    /** Extra incident edge (existing ``ABOUT`` type) so ``topic:ci-policy`` has ``degree() > 1``. */
    {
      type: 'ABOUT',
      from: 'topic:ci-policy',
      to: 'insight:b72dafa3f874480d',
    },
  )
  return JSON.stringify(j)
}

const secondGiForMerge = JSON.stringify({
  schema_version: '1.0',
  model_version: 'stub',
  prompt_version: 'v1',
  episode_id: 'ep-rfc076-e2e',
  nodes: [
    {
      id: 'episode:ep-rfc076-e2e',
      type: 'Episode',
      properties: {
        podcast_id: 'podcast:unknown',
        title: 'RFC-076 E2E second episode',
        publish_date: '2021-06-01T00:00:00Z',
      },
    },
    {
      id: 'insight:rfc076-e2e-extra',
      type: 'Insight',
      properties: {
        text: 'Extra insight for merged graph size.',
        episode_id: 'ep-rfc076-e2e',
        grounded: true,
      },
    },
    {
      id: 'quote:rfc076-e2e-extra',
      type: 'Quote',
      properties: {
        text: 'Quote body',
        episode_id: 'ep-rfc076-e2e',
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
      from: 'episode:ep-rfc076-e2e',
      to: 'insight:rfc076-e2e-extra',
    },
    {
      type: 'SUPPORTED_BY',
      from: 'insight:rfc076-e2e-extra',
      to: 'quote:rfc076-e2e-extra',
    },
    {
      type: 'ABOUT',
      from: 'insight:rfc076-e2e-extra',
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
    { id: 'person:rfc076-e2e', type: 'Person', properties: { name: 'Pat' } },
    { id: 'org:rfc076-e2e', type: 'Entity', properties: { name: 'Acme' } },
  )
  j.edges.push(
    { type: 'ABOUT', from: 'insight:b72dafa3f874480d', to: 'person:rfc076-e2e' },
    { type: 'ABOUT', from: 'insight:rfc076-e2e-degree', to: 'person:rfc076-e2e' },
    { type: 'ABOUT', from: 'insight:b72dafa3f874480d', to: 'org:rfc076-e2e' },
    { type: 'ABOUT', from: 'insight:rfc076-e2e-degree', to: 'org:rfc076-e2e' },
  )
  return JSON.stringify(j)
}

const artifactJsonPersonOrg = giJsonPersonOrgEligible()

async function mockRfc076GraphBaseline(page: Page, giArtifactBody: string = artifactJsonDegree2): Promise<void> {
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
            mtime_utc: '2024-01-01T00:00:00Z',
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

  await page.route('**/api/artifacts/metadata/ci_sample.bridge.json?**', async (route) => {
    await route.fulfill({
      status: 404,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'no bridge in e2e mock' }),
    })
  })
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
  const canvas = page.locator('.graph-canvas')
  const box = await canvas.boundingBox()
  expect(box).not.toBeNull()
  const vx = box!.x + pos!.x
  const vy = box!.y + pos!.y
  await page.mouse.move(vx, vy)
  await page.mouse.down()
  await page.mouse.up()
  await page.waitForTimeout(120)
  await page.mouse.move(vx, vy)
  await page.mouse.down()
  await page.mouse.up()
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
  const box = await page.locator('.graph-canvas').boundingBox()
  expect(box).not.toBeNull()
  const vx = box!.x + pos!.x
  const vy = box!.y + pos!.y
  await page.mouse.move(vx, vy)
  await page.mouse.down()
  await page.mouse.up()
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
  const box = await page.locator('.graph-canvas').boundingBox()
  expect(box).not.toBeNull()
  const vx = box!.x + pos!.x
  const vy = box!.y + pos!.y
  await page.keyboard.down('Shift')
  try {
    await page.mouse.move(vx, vy)
    await page.mouse.down()
    await page.mouse.up()
    await page.waitForTimeout(120)
    await page.mouse.move(vx, vy)
    await page.mouse.down()
    await page.mouse.up()
  } finally {
    await page.keyboard.up('Shift')
  }
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
  await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
  await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
  await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
  await expect(page.locator('.graph-canvas')).toBeVisible()
}

test.describe('RFC-076 graph expansion (mocked API)', () => {
  /** One graph instance per worker; serial avoids overlapping Cytoscape teardown + layout rAF races. */
  test.describe.configure({ mode: 'serial' })
  test('empty node-episodes shows truncation strip; dismiss hides it', async ({ page }) => {
    await mockRfc076GraphBaseline(page)

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
    await mockRfc076GraphBaseline(page)

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
              gi_relative_path: 'metadata/rfc076_second.gi.json',
              kg_relative_path: '',
              bridge_relative_path: '',
              episode_id: 'ep-rfc076-e2e',
            },
          ],
          truncated: true,
          total_matched: 12,
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/rfc076_second.gi.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: secondGiForMerge,
      })
    })

    await gotoGraphWithMockCorpus(page)
    await dblclickCyNode(page, 'topic:ci-policy')

    const strip = page.getByTestId('graph-expansion-truncation-line')
    await expect(strip).toBeVisible({ timeout: 15_000 })
    await expect(strip).toContainText(/Showing 1 of 12 episodes/i)
  })

  test('POST node-episodes receives topic id; expand fetches appended GI', async ({ page }) => {
    await mockRfc076GraphBaseline(page)

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
              gi_relative_path: 'metadata/rfc076_second.gi.json',
              kg_relative_path: '',
              bridge_relative_path: '',
              episode_id: 'ep-rfc076-e2e',
            },
          ],
          truncated: false,
          total_matched: null,
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/rfc076_second.gi.json?**', async (route) => {
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
    const secondGiPromise = page.waitForRequest((r) => r.url().includes('rfc076_second.gi.json'))
    await Promise.all([postPromise, secondGiPromise, dblclickCyNode(page, 'topic:ci-policy')])

    const postReq = await postPromise
    const body = postReq.postDataJSON() as { node_id?: string; path?: string; max_episodes?: number }
    expect(body.node_id).toBe('topic:ci-policy')
    expect(body.path).toBe('/mock/corpus')
    expect(body.max_episodes).toBe(RFC076_EXPAND_MAX_EPISODES)

    const giReq = await secondGiPromise
    expect(giReq.url()).toMatch(/rfc076_second\.gi\.json/i)
  })

  test('node-episodes HTTP error surfaces on truncation strip', async ({ page }) => {
    await mockRfc076GraphBaseline(page)

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
    await expect(strip).toBeVisible({ timeout: 15_000 })
    await expect(strip).toContainText(/node-episodes unavailable \(e2e\)/i)
  })

  test('expand then collapse: only one POST to node-episodes', async ({ page }) => {
    let nodeEpisodesPostCount = 0
    page.on('request', (req) => {
      if (req.url().includes('/api/corpus/node-episodes') && req.method() === 'POST') {
        nodeEpisodesPostCount += 1
      }
    })
    await mockRfc076GraphBaseline(page)

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
              gi_relative_path: 'metadata/rfc076_second.gi.json',
              kg_relative_path: '',
              bridge_relative_path: '',
              episode_id: 'ep-rfc076-e2e',
            },
          ],
          truncated: false,
          total_matched: null,
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/rfc076_second.gi.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: secondGiForMerge,
      })
    })

    await gotoGraphWithMockCorpus(page)
    const firstPost = page.waitForRequest(
      (r) => r.url().includes('/api/corpus/node-episodes') && r.method() === 'POST',
    )
    await dblclickCyNode(page, 'topic:ci-policy')
    await firstPost
    expect(nodeEpisodesPostCount).toBe(1)
    await dblclickCyNode(page, 'topic:ci-policy')
    expect(nodeEpisodesPostCount).toBe(1)
  })

  test('single activation on topic opens graph node rail (onetap)', async ({ page }) => {
    await mockRfc076GraphBaseline(page)
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

  test('Shift double-activation on topic reduces ego slice node count', async ({ page }) => {
    await mockRfc076GraphBaseline(page)
    await gotoGraphWithMockCorpus(page)
    const before = await cyNodeCount(page)
    expect(before).toBeGreaterThan(2)
    // Use **Topic** (not **Quote**): quote labels can overlap the hit target in COSE, so the
    // gesture may miss the node and skip ego without changing the node count.
    await shiftDblclickCyNode(page, 'topic:ci-policy')
    await expect.poll(async () => cyNodeCount(page)).toBeLessThan(before)
  })

  test('Episode node double-activation does not POST node-episodes', async ({ page }) => {
    let postCount = 0
    await mockRfc076GraphBaseline(page)
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
    await dblclickCyNode(page, 'episode:ci-fixture')
    expect(postCount).toBe(0)
  })

  test('Topic with degree 1 does not POST node-episodes', async ({ page }) => {
    let postCount = 0
    await mockRfc076GraphBaseline(page, artifactJsonCiSampleRaw)
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
    await dblclickCyNode(page, 'topic:ci-policy')
    expect(postCount).toBe(0)
  })

  test('Person and org nodes POST canonical ids to node-episodes', async ({ page }) => {
    const seen: string[] = []
    await mockRfc076GraphBaseline(page, artifactJsonPersonOrg)
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
    await dblclickCyNode(page, 'person:rfc076-e2e')
    await dblclickCyNode(page, 'org:rfc076-e2e')
    expect(seen).toContain('person:rfc076-e2e')
    expect(seen).toContain('org:rfc076-e2e')
  })

  test('None on corpus list resets expansion strip', async ({ page }) => {
    await mockRfc076GraphBaseline(page)
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
    await page.getByRole('navigation', { name: 'Left panel tabs' }).getByRole('button', { name: 'Corpus' }).click()
    const corpusSection = page.locator('section').filter({
      has: page.getByRole('heading', { name: 'Corpus path' }),
    })
    await corpusSection.getByRole('button', { name: 'None', exact: true }).click()
    await expect(strip).toBeHidden()
  })
})
