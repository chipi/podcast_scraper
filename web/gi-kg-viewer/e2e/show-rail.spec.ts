import { expect, test, type Page, type Route } from '@playwright/test'
import {
  liveCorpusRoot,
  liveFeeds,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
  type LiveEpisode,
  type LiveFeed,
} from './helpers'

/**
 * UXS-015 show rail (ShowRailPanel) beyond what `shows-library.spec.ts` covers: the Signals chips
 * open a topic / person in the same rail with ‹ Back to the show, "Open in graph" draws the show,
 * a second show drawn after the first lands on the graph too, Load more pages the show's episodes,
 * and an empty corpus says so.
 */

async function openShowsMode(page: Page, corpus: string): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(corpus)
  await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
  await page.getByTestId('library-mode-shows').click()
}

async function openShow(page: Page, feedId: string): Promise<void> {
  await page.getByTestId(`shows-card-${feedId}`).click()
  await expect(page.getByTestId('show-rail-panel')).toBeVisible()
}

/**
 * Episode ids drawn on the graph canvas (DEV `__GIKG_CY_DEV__` hook). Ids, not labels: the canvas
 * truncates long titles ("The Risk Panel: Diversify or…"). Merged episode nodes are
 * `__unified_ep__:<episode_id>`; an episode drawn from its KG alone (what "Open in graph" appends)
 * is `k:episode:<episode_id>`.
 */
async function graphEpisodeIds(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const cy = (
      window as unknown as {
        __GIKG_CY_DEV__?: { nodes: (s: string) => { map: (f: (n: { id: () => string }) => string) => string[] } }
      }
    ).__GIKG_CY_DEV__
    if (!cy) return []
    return cy
      .nodes('[type = "Episode"]')
      .map((n) => n.id().replace(/^(__unified_ep__:|[gk]:episode:)/, ''))
  })
}

async function expectShowOnGraph(page: Page, episodes: LiveEpisode[]): Promise<void> {
  await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
  await expect
    .poll(() => graphEpisodeIds(page), { timeout: 20_000 })
    .toEqual(expect.arrayContaining(episodes.map((e) => e.episode_id)))
}

async function liveShowEpisodes(page: Page, feed: LiveFeed): Promise<LiveEpisode[]> {
  const resp = await page.request.get(
    `/api/corpus/episodes?feed_id=${encodeURIComponent(feed.feed_id)}&limit=50`,
  )
  return ((await resp.json()) as { items: LiveEpisode[] }).items
}

test.describe('Show rail (UXS-015, live)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
    await openShowsMode(page, await liveCorpusRoot(page))
    await expect(page.getByTestId('shows-grid')).toBeVisible()
  })

  test('a Signals topic opens in the rail; ‹ Back returns to the show', async ({ page }) => {
    const feed = (await liveFeeds(page))[0]!
    await openShow(page, feed.feed_id)
    const topic = page.getByTestId('show-rail-topic').first()
    const label = (await topic.innerText()).trim()
    await topic.click()

    const rail = page.getByTestId('graph-node-detail-rail')
    await expect(rail).toBeVisible()
    await expect(rail.getByRole('heading').first()).toContainText('Topic')
    await expect(page.getByTestId('topic-entity-view')).toBeVisible()
    await expect(page.getByTestId('show-rail-panel')).toHaveCount(0)
    expect(label.length).toBeGreaterThan(0)

    await rail.getByTestId('subject-rail-back').click()
    await expect(page.getByTestId('show-rail-panel')).toContainText(feed.display_title)
  })

  test('a Signals person opens in the rail; ‹ Back returns to the show', async ({ page }) => {
    const feed = (await liveFeeds(page))[0]!
    await openShow(page, feed.feed_id)
    await page.getByTestId('show-rail-person').first().click()

    const rail = page.getByTestId('graph-node-detail-rail')
    await expect(rail.getByRole('heading').first()).toContainText('Person')
    await expect(page.getByTestId('person-landing-view')).toBeVisible()

    await rail.getByTestId('subject-rail-back').click()
    await expect(page.getByTestId('show-rail-panel')).toContainText(feed.display_title)
  })

  async function drawShow(page: Page, feed: LiveFeed): Promise<void> {
    await openShow(page, feed.feed_id)
    await expect(page.getByTestId('show-rail-episode-0')).toBeVisible()
    await page.getByTestId('show-rail-open-graph').click()
  }

  /**
   * Fails on 2026-10-05 — app defect, measured: from Shows straight to the graph (the session's
   * first Graph visit), show p01's four episodes were requested but only one was on the canvas,
   * among the 32 of the corpus lens auto-load. `ShowRailPanel.openShowInGraph` switches tab then
   * `appendRelativeArtifacts(...)`, and the first-visit corpus sync in App.vue replaces that
   * selection — the same first-visit overwrite as `artifact-list-load-graph.spec.ts`.
   */
  test('Open in graph from Shows draws every episode of the show', async ({ page }) => {
    const [a] = (await liveFeeds(page)).filter((f) => f.episode_count >= 2)
    const aEpisodes = await liveShowEpisodes(page, a!)
    await drawShow(page, a!)
    await page.waitForLoadState('networkidle')
    await expectShowOnGraph(page, aEpisodes)
  })

  test('with the Graph already open, show A then show B both land on the graph', async ({
    page,
  }) => {
    /* The topic-cluster sibling merge (POST resolve-episode-artifacts, +10 episodes after every
     * load) would otherwise put nearly the whole 40-episode corpus on the canvas after one show,
     * leaving no second show to prove anything with (measured). It is a separate feature
     * (`sibling-merge-cluster-mocks.spec.ts`); answering it with nothing isolates "Open in graph". */
    await page.route('**/api/corpus/resolve-episode-artifacts', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ resolved: [], missing_episode_ids: [] }),
      }),
    )
    // Take the first-visit auto-load out of the picture (see the test above).
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
    await page.waitForLoadState('networkidle')

    /* Pick a show with at least one episode NOT already drawn — otherwise the auto-load alone
     * satisfies the assertion (measured: the second corpus show was fully on the graph before it
     * was ever opened). */
    async function nextShowOffGraph(exclude: string[]): Promise<[LiveFeed, LiveEpisode[]]> {
      const onGraph = new Set(await graphEpisodeIds(page))
      for (const f of await liveFeeds(page)) {
        if (exclude.includes(f.feed_id)) continue
        const eps = await liveShowEpisodes(page, f)
        if (eps.some((e) => !onGraph.has(e.episode_id))) return [f, eps]
      }
      throw new Error('every corpus show is already fully on the graph; nothing to prove')
    }

    const [a, aEpisodes] = await nextShowOffGraph([])
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await drawShow(page, a)
    await expectShowOnGraph(page, aEpisodes)

    const [b, bEpisodes] = await nextShowOffGraph([a.feed_id])
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await drawShow(page, b)
    await expectShowOnGraph(page, bEpisodes)
  })
})

test.describe('Show rail paging and empty grid (mocked)', () => {
  function ep(i: number) {
    return {
      metadata_relative_path: `feeds/f1/run/metadata/e${i}.metadata.json`,
      feed_id: 'f1',
      feed_display_title: 'Mock Show',
      episode_id: `e${i}`,
      episode_title: `Mock Episode ${i}`,
      summary_preview: null,
      publish_date: `2024-01-${String(28 - (i % 28)).padStart(2, '0')}`,
    }
  }

  async function json(route: Route, body: unknown): Promise<void> {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
  }

  async function stub(page: Page, feeds: unknown[]): Promise<void> {
    await mockSignIn(page, 'creator')
    await page.route('**/api/health**', (r) =>
      json(r, { status: 'ok', corpus_library_api: true, corpus_digest_api: true }),
    )
    await page.route('**/api/corpus/feeds?**', (r) => json(r, { path: '/mock/corpus', feeds }))
    await page.route('**/api/corpus/episodes?**', (r) => {
      const cursor = new URL(r.request().url()).searchParams.get('cursor')
      return cursor === 'p2'
        ? json(r, { path: '/mock/corpus', items: [ep(51), ep(52)], next_cursor: null, total: 52 })
        : json(r, {
            path: '/mock/corpus',
            items: Array.from({ length: 50 }, (_, i) => ep(i + 1)),
            next_cursor: 'p2',
            total: 52,
          })
    })
  }

  test('Load more in the show rail appends the next page of episodes', async ({ page }) => {
    await stub(page, [{ feed_id: 'f1', display_title: 'Mock Show', episode_count: 52 }])
    await openShowsMode(page, '/mock/corpus')
    await openShow(page, 'f1')
    await expect(page.getByTestId('show-rail-episode-49')).toBeVisible()
    await expect(page.getByTestId('show-rail-episode-50')).toHaveCount(0)

    const nextPage = page.waitForRequest((r) => r.url().includes('cursor=p2'))
    await page.getByTestId('show-rail-load-more').click()
    await nextPage
    await expect(page.getByTestId('show-rail-episode-50')).toContainText('Mock Episode 51')
    await expect(page.getByTestId('show-rail-episode-51')).toContainText('Mock Episode 52')
    await expect(page.getByTestId('show-rail-load-more')).toHaveCount(0)
  })

  test('a corpus with no shows says so', async ({ page }) => {
    await stub(page, [])
    await openShowsMode(page, '/mock/corpus')
    await expect(page.getByTestId('shows-grid-empty')).toHaveText('No shows in this corpus.')
  })
})
