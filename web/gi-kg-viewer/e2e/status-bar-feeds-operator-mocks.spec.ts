import { expect, test, type Page } from '@playwright/test'
import { SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

/**
 * Browser-stubbed `/api/*` — no real `podcast serve` or corpus disk.
 *
 * With a real server, `GET /api/operator-config` may **create** `viewer_operator.yaml` with
 * `profile: cloud_balanced` when the file is missing or whitespace-only and that packaged
 * preset exists (name is hardcoded in the server). If the preset is not shipped, `content`
 * stays empty until a `PUT`.
 */

/** Match only the viewer backend path (avoid globs that also match Vite /src/api/feedsApi.ts). */
function matchExactApiPath(path: string): (url: URL) => boolean {
  return (url: URL) => url.pathname.replace(/\/$/, '') === path
}

function healthBodyWithSourcesApis() {
  return JSON.stringify({
    status: 'ok',
    corpus_library_api: true,
    corpus_digest_api: true,
    feeds_api: true,
    operator_config_api: true,
    jobs_api: true,
  })
}

async function stubCorpusPathCompanionApis(page: Page): Promise<void> {
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
  await page.route(matchExactApiPath('/api/corpus/feeds'), async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        feeds: [{ feed_id: 'f1', display_title: 'Mock Feed', episode_count: 0 }],
      }),
    })
  })
  await page.route(matchExactApiPath('/api/index/stats'), async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        available: false,
        reason: null,
        stats: null,
        reindex_recommended: false,
        rebuild_in_progress: false,
        rebuild_last_error: null,
      }),
    })
  })
  await page.route(matchExactApiPath('/api/corpus/digest'), async (route) => {
    const url = new URL(route.request().url())
    const win = url.searchParams.get('window') || 'all'
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        window: win,
        window_start_utc: '1970-01-01T00:00:00Z',
        window_end_utc: '2026-01-01T00:00:00Z',
        compact: false,
        rows: [],
        topics: [],
        topics_unavailable_reason: null,
      }),
    })
  })
}

test.describe('Status bar — feeds & operator YAML (mocked API)', () => {
  test.describe.configure({ mode: 'serial' })

  test('opens Feeds tab without calling operator-config when only feeds is opened', async ({ page }) => {
    const operatorGets: string[] = []
    await page.route(matchExactApiPath('/api/health'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: healthBodyWithSourcesApis(),
      })
    })
    await stubCorpusPathCompanionApis(page)
    await page.route(matchExactApiPath('/api/feeds'), async (route) => {
      if (route.request().method() !== 'GET') {
        await route.fulfill({ status: 405, body: 'expected GET' })
        return
      }
      const url = new URL(route.request().url())
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: url.searchParams.get('path') ?? '',
          file_relpath: 'feeds.spec.yaml',
          feeds: ['https://seed.example/rss'],
        }),
      })
    })
    await page.route(matchExactApiPath('/api/operator-config'), async (route) => {
      operatorGets.push(route.request().url())
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          corpus_path: '/mock/corpus',
          operator_config_path: '/mock/viewer_operator.yaml',
          content: 'noop: 1\n',
          available_profiles: ['cloud_balanced', 'local'],
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('status-bar-sources-trigger')).toBeVisible({ timeout: 15_000 })
    await page.getByTestId('status-bar-sources-trigger').click()
    await expect(page.getByTestId('status-bar-sources-dialog')).toBeVisible()
    await expect(page.getByTestId('sources-dialog-feeds-textarea')).toHaveValue(
      '{\n  "feeds": [\n    "https://seed.example/rss"\n  ]\n}',
    )
    await expect(page.getByTestId('sources-dialog-feeds-lines-textarea')).toBeVisible()
    await expect(page.getByTestId('sources-dialog-feeds-merge-lines')).toBeVisible()
    expect(operatorGets).toHaveLength(0)
  })

  test('Save feeds sends PUT with parsed feeds array', async ({ page }) => {
    let lastPutFeeds: unknown[] | null = null
    await page.route(matchExactApiPath('/api/health'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: healthBodyWithSourcesApis(),
      })
    })
    await stubCorpusPathCompanionApis(page)
    await page.route(matchExactApiPath('/api/feeds'), async (route) => {
      const url = new URL(route.request().url())
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            path: url.searchParams.get('path') ?? '',
            file_relpath: 'feeds.spec.yaml',
            feeds: [],
          }),
        })
        return
      }
      if (route.request().method() === 'PUT') {
        const body = route.request().postDataJSON() as { feeds?: unknown[] }
        lastPutFeeds = Array.isArray(body.feeds) ? body.feeds : []
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            path: url.searchParams.get('path') ?? '',
            file_relpath: 'feeds.spec.yaml',
            feeds: lastPutFeeds,
          }),
        })
      }
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('status-bar-sources-trigger')).toBeVisible({ timeout: 15_000 })
    await page.getByTestId('status-bar-sources-trigger').click()
    await page
      .getByTestId('sources-dialog-feeds-textarea')
      .fill('{\n  "feeds": [\n    "  https://a.example/x  ",\n    "https://b.example/y"\n  ]\n}')
    await page.getByRole('button', { name: 'Save feeds' }).click()
    expect(lastPutFeeds).toEqual(['  https://a.example/x  ', 'https://b.example/y'])
  })

  test('Append lines merges URLs into feeds JSON without save', async ({ page }) => {
    await page.route(matchExactApiPath('/api/health'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: healthBodyWithSourcesApis(),
      })
    })
    await stubCorpusPathCompanionApis(page)
    await page.route(matchExactApiPath('/api/feeds'), async (route) => {
      const url = new URL(route.request().url())
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: url.searchParams.get('path') ?? '',
          file_relpath: 'feeds.spec.yaml',
          feeds: ['https://existing.example/rss'],
        }),
      })
    })
    await page.route(matchExactApiPath('/api/operator-config'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          corpus_path: '/mock/corpus',
          operator_config_path: '/mock/viewer_operator.yaml',
          content: 'noop: 1\n',
          available_profiles: ['cloud_balanced'],
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await page.getByTestId('status-bar-sources-trigger').click()
    await page.getByTestId('sources-dialog-feeds-lines-textarea').fill('https://new.example/feed\n')
    await page.getByTestId('sources-dialog-feeds-merge-lines').click()
    const v = await page.getByTestId('sources-dialog-feeds-textarea').inputValue()
    expect(v).toContain('https://existing.example/rss')
    expect(v).toContain('https://new.example/feed')
  })

  test('Operator tab loads YAML and save sends PUT body', async ({ page }) => {
    let lastPutContent: string | null = null
    await page.route(matchExactApiPath('/api/health'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: healthBodyWithSourcesApis(),
      })
    })
    await stubCorpusPathCompanionApis(page)
    await page.route(matchExactApiPath('/api/feeds'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          file_relpath: 'feeds.spec.yaml',
          feeds: [],
        }),
      })
    })
    await page.route(matchExactApiPath('/api/operator-config'), async (route) => {
      const url = new URL(route.request().url())
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            corpus_path: url.searchParams.get('path') ?? '',
            operator_config_path: '/mock/custom.yaml',
            content: 'keep: true\n',
            available_profiles: ['cloud_balanced', 'local'],
          }),
        })
        return
      }
      if (route.request().method() === 'PUT') {
        const body = route.request().postDataJSON() as { content?: string }
        lastPutContent = typeof body.content === 'string' ? body.content : null
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            corpus_path: url.searchParams.get('path') ?? '',
            operator_config_path: '/mock/custom.yaml',
            content: lastPutContent ?? '',
            available_profiles: ['cloud_balanced', 'local'],
          }),
        })
      }
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('status-bar-sources-trigger')).toBeVisible({
      timeout: 15_000,
    })
    await page.getByTestId('status-bar-sources-trigger').click()
    await page.getByTestId('sources-dialog-tab-operator').click()
    await expect(page.getByTestId('sources-dialog-profile-select')).toBeVisible()
    await expect(page.getByTestId('sources-dialog-operator-textarea')).toHaveValue('keep: true')
    await page.getByTestId('sources-dialog-operator-textarea').fill('keep: true\nextra: 2\n')
    await page.getByRole('button', { name: 'Save YAML' }).click()
    expect(lastPutContent).toBe('keep: true\nextra: 2\n')
  })

  test('health dialog lists feeds, operator, and pipeline jobs API rows as Yes', async ({ page }) => {
    await page.route(matchExactApiPath('/api/health'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: healthBodyWithSourcesApis(),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await page.getByTestId('status-bar-health-trigger').click()
    const healthDialog = page.locator('[aria-labelledby="status-bar-health-dialog-title"]')
    await expect(healthDialog.getByText('Feeds file API')).toBeVisible()
    await expect(healthDialog.getByText('Operator YAML API')).toBeVisible()
    await expect(healthDialog.getByText('Pipeline jobs API')).toBeVisible()
    const feedsDt = healthDialog.locator('dt').filter({ hasText: /Feeds file API/ })
    await expect(feedsDt.locator('xpath=./following-sibling::dd[1]')).toHaveText('Yes')
    const opDt = healthDialog.locator('dt').filter({ hasText: /Operator YAML API/ })
    await expect(opDt.locator('xpath=./following-sibling::dd[1]')).toHaveText('Yes')
    const jobsDt = healthDialog.locator('dt').filter({ hasText: /Pipeline jobs API/ })
    await expect(jobsDt.locator('xpath=./following-sibling::dd[1]')).toHaveText('Yes')
  })
})
