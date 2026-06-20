import { expect, test, type Page } from '@playwright/test'
import { SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

/**
 * #694 — per-feed override drill-in. Configuring a feed sets structured override
 * fields (e.g. `max_episodes`) on that entry only; other feeds round-trip
 * unchanged, and the result persists via `PUT /api/feeds`.
 */
function matchExactApiPath(path: string): (url: URL) => boolean {
  return (url: URL) => url.pathname.replace(/\/$/, '') === path
}

async function stubCompanionApis(page: Page): Promise<void> {
  await page.route(
    (url) => {
      const p = url.pathname.replace(/\/$/, '')
      return p === '/api/artifacts' || p.startsWith('/api/artifacts/')
    },
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [], hints: [] }),
      })
    },
  )
  await page.route(matchExactApiPath('/api/index/stats'), async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        available: false,
        stats: null,
        reindex_recommended: false,
        rebuild_in_progress: false,
        rebuild_last_error: null,
      }),
    })
  })
}

test.describe('Per-feed override editor (#694)', () => {
  test('Configure sets max_episodes on one feed and persists via PUT', async ({ page }) => {
    let lastPutFeeds: unknown[] | null = null

    await page.route(matchExactApiPath('/api/health'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          feeds_api: true,
          operator_config_api: true,
        }),
      })
    })
    await stubCompanionApis(page)
    await page.route(matchExactApiPath('/api/feeds'), async (route) => {
      const method = route.request().method()
      if (method === 'PUT') {
        const body = route.request().postDataJSON() as { feeds: unknown[] }
        lastPutFeeds = body.feeds
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            path: '/mock/corpus',
            file_relpath: 'feeds.spec.yaml',
            feeds: body.feeds,
          }),
        })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          file_relpath: 'feeds.spec.yaml',
          feeds: ['https://a.example/rss', 'https://b.example/rss'],
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await page.getByTestId('status-bar-sources-trigger').click()
    await expect(page.getByTestId('status-bar-sources-dialog')).toBeVisible()
    await expect(page.getByTestId('sources-dialog-feeds-row-0')).toContainText('https://a.example/rss')

    // Drill into feed 0 and set a per-feed override.
    await page.getByTestId('sources-dialog-feeds-row-configure-0').click()
    await expect(page.getByTestId('feed-override-editor')).toBeVisible()
    await expect(page.getByTestId('feed-override-url')).toHaveText('https://a.example/rss')
    await page.getByTestId('feed-override-max-episodes').fill('2')
    await page.getByTestId('feed-override-save').click()

    // Back to the list; PUT carried the override on feed 0 only.
    await expect(page.getByTestId('feed-override-editor')).toBeHidden()
    await expect.poll(() => lastPutFeeds).not.toBeNull()
    expect(lastPutFeeds).toEqual([
      { url: 'https://a.example/rss', max_episodes: 2 },
      'https://b.example/rss',
    ])
  })
})
