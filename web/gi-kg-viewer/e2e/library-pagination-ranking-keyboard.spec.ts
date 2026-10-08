import { expect, test, type Page, type Route } from '@playwright/test'
import {
  liveCorpusRoot,
  liveDigestRows,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
  type LiveEpisode,
} from './helpers'

/**
 * Library list mechanics (UXS-003 / PRD-033): cursor pagination with **Load more**, the heading
 * count, FR2.2 "matched episodes float to the top" under an active search context, and the
 * roving keyboard on Library and Digest rows that drives the Episode rail.
 */

function episodeRail(page: Page) {
  return page.getByRole('region', { name: 'Episode', exact: true })
}

function ep(id: string, title: string, publishDate: string) {
  return {
    metadata_relative_path: `feeds/f1/run/metadata/${id}.metadata.json`,
    feed_id: 'f1',
    feed_display_title: 'Mock Feed',
    episode_id: id,
    episode_title: title,
    summary_title: null,
    summary_bullets_preview: [],
    summary_preview: `${title} summary`,
    publish_date: publishDate,
  }
}

async function json(route: Route, body: unknown): Promise<void> {
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
}

async function stubLibrary(page: Page, episodes: (r: URL) => unknown): Promise<void> {
  await mockSignIn(page, 'creator')
  await page.route('**/api/health**', (r) =>
    json(r, { status: 'ok', corpus_library_api: true, corpus_digest_api: true, search_api: true }),
  )
  await page.route('**/api/corpus/feeds?**', (r) =>
    json(r, {
      path: '/mock/corpus',
      feeds: [{ feed_id: 'f1', display_title: 'Mock Feed', episode_count: 4 }],
    }),
  )
  await page.route('**/api/corpus/episodes?**', (r) => json(r, episodes(new URL(r.request().url()))))
}

async function openLibrary(page: Page, corpus: string): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(corpus)
  await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
  await expect(page.getByTestId('library-root')).toBeVisible()
}

const rows = (page: Page) => page.locator('[data-library-episode-row]')
const heading = (page: Page) => page.locator('#library-episodes-heading')

test.describe('Library pagination and FR2.2 ranking (mocked)', () => {
  test('a server without `total`: heading reads (N+), Load more appends the next page', async ({
    page,
  }) => {
    // A full first page (20) so the list overflows: a short page leaves the infinite-scroll
    // sentinel on screen and the second page loads by itself before Load more can be pressed.
    const first = Array.from({ length: 20 }, (_, i) =>
      ep(`e${i + 1}`, `Episode ${i + 1}`, `2024-02-${String(28 - i).padStart(2, '0')}`),
    )
    const pages: Record<string, unknown> = {
      '': { path: '/mock/corpus', items: first, next_cursor: 'c2' },
      c2: {
        path: '/mock/corpus',
        items: [ep('e21', 'Late One', '2024-01-02'), ep('e22', 'Late Two', '2024-01-01')],
        next_cursor: null,
      },
    }
    await stubLibrary(page, (u) => pages[u.searchParams.get('cursor') ?? ''])
    await openLibrary(page, '/mock/corpus')

    await expect(rows(page)).toHaveCount(20)
    await expect(heading(page)).toHaveText(/Episodes\s*\(20\+\)/)
    const loadMore = page.getByRole('button', { name: 'Load more' })
    await loadMore.click()
    await expect(rows(page)).toHaveCount(22)
    await expect(rows(page).nth(20)).toHaveAccessibleName('Late One, Mock Feed')
    await expect(rows(page).nth(21)).toHaveAccessibleName('Late Two, Mock Feed')
    await expect(heading(page)).toHaveText(/Episodes\s*\(22\)/)
    await expect(loadMore).toHaveCount(0)
  })

  test('after a search, matched episodes come first (by score) with a "Why this episode" line', async ({
    page,
  }) => {
    await stubLibrary(page, () => ({
      path: '/mock/corpus',
      items: [
        ep('e1', 'Alpha', '2024-03-03'),
        ep('e2', 'Bravo', '2024-02-02'),
        ep('e3', 'Charlie', '2024-01-01'),
      ],
      next_cursor: null,
      total: 3,
    }))
    const hit = (id: string, score: number, text: string) => ({
      doc_id: `insight:${id}`,
      score,
      text,
      metadata: { doc_type: 'insight', episode_id: id, episode_title: id, feed_id: 'f1' },
    })
    await page.route('**/api/search?**', (r) =>
      json(r, {
        query: 'ranking probe',
        results: [hit('e3', 0.9, 'Charlie says the key thing.'), hit('e1', 0.4, 'Alpha mentions it.')],
        lift_stats: null,
        error: null,
        detail: null,
      }),
    )
    await openLibrary(page, '/mock/corpus')
    await expect(rows(page).first()).toHaveAccessibleName('Alpha, Mock Feed')

    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await page.locator('#search-q').fill('ranking probe')
    await page.locator('#search-q').press('Enter')
    await expect(page.getByTestId('search-workspace').locator('article').first()).toBeVisible()

    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(rows(page)).toHaveCount(3)
    await expect(rows(page).nth(0)).toHaveAccessibleName('Charlie, Mock Feed')
    await expect(rows(page).nth(1)).toHaveAccessibleName('Alpha, Mock Feed')
    await expect(rows(page).nth(2)).toHaveAccessibleName('Bravo, Mock Feed')
    await expect(rows(page).nth(0).getByTestId('library-row-why')).toContainText(
      'Charlie says the key thing.',
    )
    await expect(rows(page).nth(2).getByTestId('library-row-why')).toHaveCount(0)
  })
})

test.describe('Library pagination and row keyboard (live)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  async function firstPage(page: Page): Promise<LiveEpisode[]> {
    const root = await liveCorpusRoot(page)
    const resp = await page.request.get(
      `/api/corpus/episodes?${new URLSearchParams({ path: root, limit: '20' })}`,
    )
    return ((await resp.json()) as { items: LiveEpisode[] }).items
  }

  test('heading reads (20 of N); Load more appends the next page', async ({ page }) => {
    await openLibrary(page, await liveCorpusRoot(page))
    await expect(rows(page)).toHaveCount(20)
    const m = /\(20 of (\d+)\)/.exec(await heading(page).innerText())
    expect(m, 'heading shows "(20 of N)" against a server that returns total').not.toBeNull()
    const total = Number(m![1])
    expect(total).toBeGreaterThan(20)

    // Dispatched, not clicked: a real click scrolls the button into view, which also brings the
    // scroll-to-load sentinel just above it into view. That loads the page on its own, re-renders
    // the button as "Loading…" mid-click, and can keep loading pages until the button is gone —
    // the click then waits out the whole test timeout. This test is about the button.
    await page.getByRole('button', { name: 'Load more' }).dispatchEvent('click')
    await expect(rows(page)).toHaveCount(Math.min(40, total))
  })

  test('ArrowDown / End / Home on Library rows move the Episode rail', async ({ page }) => {
    await openLibrary(page, await liveCorpusRoot(page))
    const items = await firstPage(page)
    await rows(page).first().click()
    await expect(episodeRail(page).getByRole('heading', { name: items[0]!.episode_title })).toBeVisible()

    await page.keyboard.press('ArrowDown')
    await expect(episodeRail(page).getByRole('heading', { name: items[1]!.episode_title })).toBeVisible()
    await expect(rows(page).nth(1)).toBeFocused()

    await page.keyboard.press('End')
    const last = items[items.length - 1]!
    await expect(episodeRail(page).getByRole('heading', { name: last.episode_title })).toBeVisible()

    await page.keyboard.press('Home')
    await expect(episodeRail(page).getByRole('heading', { name: items[0]!.episode_title })).toBeVisible()
    await expect(rows(page).first()).toBeFocused()
  })

  test('ArrowDown / End / Home on Digest Recent rows move the Episode rail', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    const digestRows = page.locator('[data-digest-recent-row]')
    await expect(digestRows.first()).toBeVisible()
    const n = await digestRows.count()
    expect(n).toBeGreaterThan(2)
    const titles = (await liveDigestRows(page)).map((r) => r.episode_title)

    await digestRows.first().click()
    await expect(episodeRail(page).getByRole('heading', { name: titles[0]! })).toBeVisible()
    await page.keyboard.press('ArrowDown')
    await expect(episodeRail(page).getByRole('heading', { name: titles[1]! })).toBeVisible()
    await expect(digestRows.nth(1)).toBeFocused()
    await page.keyboard.press('End')
    await expect(digestRows.nth(n - 1)).toBeFocused()
    await expect(episodeRail(page).getByRole('heading', { name: titles[n - 1]! })).toBeVisible()
    await page.keyboard.press('Home')
    await expect(digestRows.first()).toBeFocused()
    await expect(episodeRail(page).getByRole('heading', { name: titles[0]! })).toBeVisible()
  })
})
