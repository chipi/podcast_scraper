import { expect, test, type Page } from '@playwright/test'
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
 * UXS-015 / RFC-104 — the operator Library tab's shows-first browse.
 *
 * Library → Shows mode → shows grid → open a show → its episode list → click an episode → the
 * episode opens in the subject rail (the same focusEpisode handoff a flat-Library row uses).
 * Shows is opt-in (Episodes is the default, PRD-044 OQ1), so each test toggles first.
 *
 * #1619 — fully migrated to the live API. Every endpoint this flow needs
 * (`/api/corpus/{feeds,episodes,episodes/detail,feed-signals}`) serves real data for the v3
 * fixture corpus, so the show under test is whichever feed the server lists first rather than a
 * hand-authored `Alpha Show`.
 */

/** The feed the grid renders first, plus its episodes and signals, all read from the server. */
async function liveShowSubject(page: Page): Promise<{
  feed: LiveFeed
  episodes: LiveEpisode[]
  topicCount: number
  people: { name: string }[]
}> {
  const feeds = await liveFeeds(page)
  /* Pick a feed with at least two episodes — the rail asserts `show-rail-episode-0` and `-1`. */
  const feed = feeds.find((f) => f.episode_count >= 2) ?? feeds[0]!
  const epResp = await page.request.get(
    `/api/corpus/episodes?feed_id=${encodeURIComponent(feed.feed_id)}&limit=5`,
  )
  const { items } = (await epResp.json()) as { items: LiveEpisode[] }
  const sigResp = await page.request.get(
    `/api/corpus/feed-signals?feed_id=${encodeURIComponent(feed.feed_id)}`,
  )
  const signals = (await sigResp.json()) as {
    top_topics: { label: string }[]
    key_people: { name: string }[]
  }
  return {
    feed,
    episodes: items,
    topicCount: signals.top_topics.length,
    people: signals.key_people,
  }
}

test.describe('Operator Shows Library (shows-first browse)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  async function openShowsMode(page: Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-mode-shows').click()
    await expect(page.getByTestId('shows-grid')).toBeVisible()
  }

  test('grid lists shows; opening one opens the show in the RIGHT RAIL (grid stays)', async ({
    page,
  }) => {
    await openShowsMode(page)
    const { feed, episodes } = await liveShowSubject(page)
    const others = (await liveFeeds(page)).filter((f) => f.feed_id !== feed.feed_id)

    // The corpus's shows render as cards, each carrying its own episode count.
    await expect(page.getByTestId(`shows-card-${feed.feed_id}`)).toBeVisible()
    await expect(page.getByTestId(`shows-card-${others[0]!.feed_id}`)).toBeVisible()
    await expect(page.getByTestId(`shows-card-${feed.feed_id}`)).toContainText(
      `${feed.episode_count} episodes`,
    )

    // Open the show → it opens in the RIGHT SUBJECT RAIL (ShowRailPanel), not in-panel; the grid
    // stays put in the main library surface. Header + feed-scoped episodes.
    const scopedReq = page.waitForRequest(
      (r) =>
        r.url().includes('/api/corpus/episodes') &&
        r.url().includes(`feed_id=${feed.feed_id}`),
    )
    await page.getByTestId(`shows-card-${feed.feed_id}`).click()
    await scopedReq
    const rail = page.getByTestId('show-rail-panel')
    await expect(rail).toBeVisible()
    await expect(rail).toContainText(feed.display_title)
    await expect(page.getByTestId('show-rail-episode-0')).toContainText(
      episodes[0]!.episode_title,
    )
    await expect(page.getByTestId('show-rail-episode-1')).toContainText(
      episodes[1]!.episode_title,
    )
    // The grid remains — the show opened in the rail, not the same surface.
    await expect(page.getByTestId('shows-grid')).toBeVisible()
  })

  test('the show rail shows a Signals band (top topics + key people)', async ({ page }) => {
    await openShowsMode(page)
    const { feed, topicCount, people } = await liveShowSubject(page)
    await page.getByTestId(`shows-card-${feed.feed_id}`).click()
    await expect(page.getByTestId('show-rail-panel')).toBeVisible()

    const signals = page.getByTestId('show-rail-signals')
    await expect(signals).toBeVisible()
    /* Counts come from the server's own signals payload — a re-enriched corpus changes how many
     * topics a show has without changing what this test is about. */
    expect(topicCount).toBeGreaterThan(0)
    expect(people.length).toBeGreaterThan(0)
    await expect(page.getByTestId('show-rail-topic')).toHaveCount(topicCount)
    await expect(page.getByTestId('show-rail-person')).toHaveCount(people.length)
    await expect(page.getByTestId('show-rail-person').first()).toContainText(people[0]!.name)
  })

  test('episode in the show rail opens in the same rail, with ‹ Back to the show', async ({
    page,
  }) => {
    await openShowsMode(page)
    const { feed, episodes } = await liveShowSubject(page)
    await page.getByTestId(`shows-card-${feed.feed_id}`).click()
    await expect(page.getByTestId('show-rail-episode-0')).toBeVisible()

    // Click an episode → it opens in the same rail (focusEpisode); the show rail is replaced.
    await page.getByTestId('show-rail-episode-0').click()
    const episodeRegion = page.getByRole('region', { name: 'Episode', exact: true })
    await expect(
      episodeRegion.getByRole('heading', { name: episodes[0]!.episode_title }),
    ).toBeVisible()

    // ‹ Back returns to the show rail (subject history stack).
    await episodeRegion.getByTestId('subject-rail-back').click()
    await expect(page.getByTestId('show-rail-panel')).toContainText(feed.display_title)
  })

  test('closing the show rail leaves the grid in place', async ({ page }) => {
    await openShowsMode(page)
    const { feed } = await liveShowSubject(page)
    await page.getByTestId(`shows-card-${feed.feed_id}`).click()
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
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    // Default = Episodes → the flat LibraryView is shown, not the shows grid.
    await expect(page.getByTestId('library-root')).toBeVisible()
    await expect(page.getByTestId('shows-grid')).toHaveCount(0)
  })
})
