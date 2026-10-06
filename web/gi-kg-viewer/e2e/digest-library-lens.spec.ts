import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * Digest and Library share ONE corpus lens ("Published on or after", VIEWER_IA): setting the
 * date on either tab is the date on the other, and Digest's "Search topic" carries it into
 * Search's Since chip.
 */

function isDigestRequest(since: string | null) {
  return (r: { url(): string }): boolean => {
    const u = new URL(r.url())
    return u.pathname === '/api/corpus/digest' && u.searchParams.get('since') === since
  }
}

test.describe('Digest ↔ Library shared date lens (live)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await expect(page.getByTestId('digest-root')).toBeVisible()
  })

  test('a date set on Digest is the Library date, and back', async ({ page }) => {
    const digestChip = page.getByTestId('digest-chip-date')
    await expect(digestChip).toHaveText('Date ▾')
    await digestChip.click()
    await page.getByTestId('digest-popover-date').getByRole('button', { name: '30d' }).click()
    await expect(digestChip).toHaveText('Date: Last 30d ▾')

    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    const libraryChip = page.getByTestId('library-chip-date')
    await expect(libraryChip).toHaveText('Date: Last 30d ▾')

    await libraryChip.click()
    const custom = page.getByTestId('library-chip-date-custom')
    await custom.fill('2025-01-01')
    await custom.press('Enter')
    await expect(libraryChip).toHaveText('Date: ≥ 2025-01-01 ▾')

    // Back on Digest the chip shows the Library's date AND the digest is fetched for it.
    const refetch = page.waitForRequest(isDigestRequest('2025-01-01'))
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await expect(digestChip).toHaveText('Date: ≥ 2025-01-01 ▾')
    await refetch
  })
})

test.describe('Digest lens → Search, and a server without the digest route (mocked)', () => {
  const HIT = {
    metadata_relative_path: 'metadata/ep1.metadata.json',
    episode_title: 'Digest Episode Alpha',
    feed_id: 'f1',
    feed_display_title: 'Mock Feed Show',
    publish_date: '2024-06-05',
    score: 0.91,
    summary_preview: 'Digest summary',
    episode_id: 'e1',
    gi_relative_path: 'metadata/ep1.gi.json',
    kg_relative_path: 'metadata/ep1.kg.json',
    has_gi: true,
    has_kg: false,
  }

  async function stub(page: Page, opts: { digestApi: boolean }): Promise<void> {
    await mockSignIn(page, 'creator')
    await page.route('**/api/health**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: opts.digestApi,
        }),
      }),
    )
    await page.route('**/api/corpus/digest**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          window: 'since',
          window_start_utc: '2024-01-01T00:00:00Z',
          window_end_utc: '2024-06-08T00:00:00Z',
          compact: false,
          rows: [],
          topics: [
            {
              topic_id: 't1',
              label: 'Mock Topic Band',
              query: 'climate science',
              graph_topic_id: 'topic:mock-topic-band',
              hits: [HIT],
            },
          ],
          topics_unavailable_reason: null,
        }),
      }),
    )
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
  }

  test('"Search topic" with a date lens set carries it into the Search Since chip', async ({
    page,
  }) => {
    await stub(page, { digestApi: true })
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await page.getByTestId('digest-chip-date').click()
    const custom = page.getByTestId('digest-chip-date-custom')
    await custom.fill('2024-01-01')
    await custom.press('Enter')
    await expect(page.getByTestId('digest-chip-date')).toHaveText('Date: ≥ 2024-01-01 ▾')

    await page.getByRole('button', { name: 'Search topic' }).first().click()
    await expect(page.getByTestId('search-workspace')).toBeVisible()
    await expect(page.locator('#search-q')).toHaveValue('climate science')
    await expect(page.getByTestId('search-chip-since')).toHaveText('Since: ≥ 2024-01-01 ▾')
  })

  test('health without corpus_digest_api explains the API needs upgrading', async ({ page }) => {
    const digestCalls: string[] = []
    page.on('request', (r) => {
      if (new URL(r.url()).pathname === '/api/corpus/digest') digestCalls.push(r.url())
    })
    await stub(page, { digestApi: false })
    const root = page.getByTestId('digest-root')
    await expect(root).toBeVisible()
    await expect(root).toContainText(
      'This API build does not expose the digest endpoint (GET /api/corpus/digest).',
    )
    await expect(root).toContainText(
      'Upgrade the viewer API process; Library may still work if the catalog is available.',
    )
    // ...and does not call the route it was just told is missing.
    await page.waitForLoadState('networkidle')
    expect(digestCalls).toEqual([])
  })
})
