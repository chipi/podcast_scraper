import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * UXS-015 / RFC-104 — the operator Library tab's shows-first browse.
 *
 * Drives the full flow against mocked /api/corpus endpoints: Library → Shows mode
 * → shows grid → open a show → its episode list → click an episode → the episode
 * opens in the subject rail (the same focusEpisode handoff a flat-Library row uses).
 * Shows is opt-in (Episodes is the default, PRD-044 OQ1), so each test toggles first.
 */

const FEEDS = {
  path: '/mock/corpus',
  feeds: [
    {
      feed_id: 'alpha',
      display_title: 'Alpha Show',
      episode_count: 2,
      description: 'The alpha podcast about everything.',
      image_url: 'http://mock/alpha.jpg',
    },
    { feed_id: 'beta', display_title: 'Beta Show', episode_count: 1 },
  ],
}

const ALPHA_EPISODES = {
  path: '/mock/corpus',
  feed_id: 'alpha',
  items: [
    {
      metadata_relative_path: 'metadata/a1.metadata.json',
      feed_id: 'alpha',
      feed_display_title: 'Alpha Show',
      summary_title: 'First alpha',
      summary_preview: 'First alpha — a point',
      episode_id: 'a1',
      episode_title: 'Alpha Episode One',
      publish_date: '2026-06-01',
      has_gi: true,
      has_kg: true,
    },
    {
      metadata_relative_path: 'metadata/a2.metadata.json',
      feed_id: 'alpha',
      feed_display_title: 'Alpha Show',
      episode_id: 'a2',
      episode_title: 'Alpha Episode Two',
      publish_date: '2026-05-01',
    },
  ],
  next_cursor: null,
}

const ALPHA_SIGNALS = {
  path: '/mock/corpus',
  feed_id: 'alpha',
  episode_count: 2,
  top_topics: [
    { topic_id: 'topic:ai', label: 'AI', episode_count: 2 },
    { topic_id: 'topic:ethics', label: 'Ethics', episode_count: 1 },
  ],
  key_people: [{ person_id: 'person:jane', name: 'Jane Doe', episode_count: 2 }],
}

test.describe('Operator Shows Library (shows-first browse)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok', corpus_library_api: true, corpus_digest_api: true }),
      })
    })
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
      })
    })
    await page.route('**/api/corpus/feeds**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(FEEDS),
      })
    })
    await page.route('**/api/corpus/episodes/detail**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          metadata_relative_path: 'metadata/a1.metadata.json',
          feed_id: 'alpha',
          episode_id: 'a1',
          episode_title: 'Alpha Episode One',
          publish_date: '2026-06-01',
          summary_title: 'First alpha',
          summary_bullets: ['a point'],
          summary_text: 'Prose summary.',
          gi_relative_path: 'metadata/a1.gi.json',
          kg_relative_path: 'metadata/a1.kg.json',
          has_gi: true,
          has_kg: true,
          cil_digest_topics: [],
        }),
      })
    })
    await page.route('**/api/corpus/feed-signals**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(ALPHA_SIGNALS),
      })
    })
    // Feed-scoped episode list (the show detail fetch).
    await page.route('**/api/corpus/episodes?**', async (route, request) => {
      // Only the show-detail (feed_id=alpha) list is asserted; default returns alpha.
      if (request.url().includes('feed_id=alpha') || !request.url().includes('feed_id=')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(ALPHA_EPISODES),
        })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', feed_id: null, items: [], next_cursor: null }),
      })
    })
    await page.route('**/api/corpus/episodes/similar**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          source_metadata_relative_path: 'metadata/a1.metadata.json',
          query_used: '',
          items: [],
          error: null,
          detail: null,
        }),
      })
    })
    await page.route('**/api/index/stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ available: false, reason: 'no_index', stats: null, reindex_recommended: false }),
      })
    })
  })

  async function openShowsMode(page: import('@playwright/test').Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-mode-shows').click()
    await expect(page.getByTestId('shows-grid')).toBeVisible()
  }

  test('grid lists shows; opening one opens the show in the RIGHT RAIL (grid stays)', async ({
    page,
  }) => {
    await openShowsMode(page)

    // Both shows render as cards.
    await expect(page.getByTestId('shows-card-alpha')).toBeVisible()
    await expect(page.getByTestId('shows-card-beta')).toBeVisible()
    await expect(page.getByTestId('shows-card-alpha')).toContainText('2 episodes')

    // Open Alpha → the show opens in the RIGHT SUBJECT RAIL (ShowRailPanel), not in-panel;
    // the grid stays put in the main library surface. Header + feed-scoped episodes.
    const scopedReq = page.waitForRequest(
      (r) => r.url().includes('/api/corpus/episodes') && r.url().includes('feed_id=alpha'),
    )
    await page.getByTestId('shows-card-alpha').click()
    await scopedReq
    const rail = page.getByTestId('show-rail-panel')
    await expect(rail).toBeVisible()
    await expect(rail).toContainText('Alpha Show')
    await expect(page.getByTestId('show-rail-episode-0')).toContainText('Alpha Episode One')
    await expect(page.getByTestId('show-rail-episode-1')).toContainText('Alpha Episode Two')
    // The grid remains — the show opened in the rail, not the same surface.
    await expect(page.getByTestId('shows-grid')).toBeVisible()
  })

  test('the show rail shows a Signals band (top topics + key people)', async ({ page }) => {
    await openShowsMode(page)
    await page.getByTestId('shows-card-alpha').click()
    await expect(page.getByTestId('show-rail-panel')).toBeVisible()

    const signals = page.getByTestId('show-rail-signals')
    await expect(signals).toBeVisible()
    const topics = page.getByTestId('show-rail-topic')
    await expect(topics).toHaveCount(2)
    await expect(topics.first()).toContainText('AI')
    const people = page.getByTestId('show-rail-person')
    await expect(people).toHaveCount(1)
    await expect(people.first()).toContainText('Jane Doe')
  })

  test('episode in the show rail opens in the same rail, with ‹ Back to the show', async ({
    page,
  }) => {
    await openShowsMode(page)
    await page.getByTestId('shows-card-alpha').click()
    await expect(page.getByTestId('show-rail-episode-0')).toBeVisible()

    // Click an episode → it opens in the same rail (focusEpisode); the show rail is replaced.
    await page.getByTestId('show-rail-episode-0').click()
    const episodeRegion = page.getByRole('region', { name: 'Episode', exact: true })
    await expect(episodeRegion.getByRole('heading', { name: 'Alpha Episode One' })).toBeVisible()

    // ‹ Back returns to the show rail (subject history stack).
    await episodeRegion.getByTestId('subject-rail-back').click()
    await expect(page.getByTestId('show-rail-panel')).toContainText('Alpha Show')
  })

  test('closing the show rail leaves the grid in place', async ({ page }) => {
    await openShowsMode(page)
    await page.getByTestId('shows-card-alpha').click()
    await expect(page.getByTestId('show-rail-panel')).toBeVisible()

    await page.getByTestId('show-detail-rail').getByTestId('subject-rail-close').click()
    await expect(page.getByTestId('show-rail-panel')).toHaveCount(0)
    await expect(page.getByTestId('shows-grid')).toBeVisible()
  })

  test('mode is remembered: Episodes remains the default until Shows is chosen', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    // Default = Episodes → the flat LibraryView is shown, not the shows grid.
    await expect(page.getByTestId('library-root')).toBeVisible()
    await expect(page.getByTestId('shows-grid')).toHaveCount(0)
  })
})
