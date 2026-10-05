import { expect, test, type Page, type Route } from '@playwright/test'
import { mainViewsNav, mockSignIn, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

/**
 * Search v3 §S4 operator bar — the edge states `search-operator-bar.spec.ts` (live) cannot reach:
 * a hit set with nothing graph-resolvable, a failing operator call, and operator responses that
 * come back empty. Each is a response shape, so `/api/search` is mocked and branches on
 * `operator=`.
 */

const QUERY = 'operator edges'

/** Hits with no episode_id / source_id — nothing the On-graph chip could focus. */
const ID_LESS_HITS = [
  { doc_id: 'transcript:1', score: 0.7, text: 'Hit one', metadata: { doc_type: 'transcript' } },
  { doc_id: 'transcript:2', score: 0.6, text: 'Hit two', metadata: { doc_type: 'transcript' } },
]

type OperatorReply = { status: number; body: unknown } | null

async function openWithOperatorReplies(
  page: Page,
  replies: { cluster?: OperatorReply; consensus?: OperatorReply },
): Promise<void> {
  await mockSignIn(page, 'creator')
  await page.route('**/api/health**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'ok', corpus_library_api: true, search_api: true }),
    }),
  )
  await page.route('**/api/search?**', (route: Route) => {
    const op = new URL(route.request().url()).searchParams.get('operator') as
      | 'cluster'
      | 'consensus'
      | null
    const reply = op ? replies[op] : null
    if (reply) {
      return route.fulfill({
        status: reply.status,
        contentType: reply.status === 200 ? 'application/json' : 'text/plain',
        body: typeof reply.body === 'string' ? reply.body : JSON.stringify(reply.body),
      })
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ query: QUERY, results: ID_LESS_HITS, error: null, detail: null }),
    })
  })
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill('/mock/corpus')
  await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
  await page.locator('#search-q').fill(QUERY)
  await page.locator('#search-q').press('Enter')
  await expect(page.getByTestId('result-set-operator-bar')).toBeVisible()
}

const okEmpty = (extra: Record<string, unknown>) => ({
  status: 200,
  body: { query: QUERY, results: ID_LESS_HITS, error: null, detail: null, ...extra },
})

test.describe('Search operator bar — edge states (mocked)', () => {
  test('a hit set with no graph ids disables "On graph (no ids)"', async ({ page }) => {
    await openWithOperatorReplies(page, {})
    const chip = page.getByTestId('operator-chip-graph')
    await expect(chip).toHaveText('On graph (no ids)')
    await expect(chip).toBeDisabled()
  })

  test('Cluster with no clusters shows the empty line', async ({ page }) => {
    await openWithOperatorReplies(page, { cluster: okEmpty({ clusters: [] }) })
    await page.getByTestId('operator-chip-cluster').click()
    await expect(page.getByTestId('operator-cluster-panel')).toBeVisible()
    await expect(page.getByTestId('operator-cluster-empty')).toHaveText(
      'No clusters — no hit resolves to a topic or theme cluster surface.',
    )
    await expect(page.getByTestId('operator-cluster-list')).toHaveCount(0)
    await expect(page.getByTestId('operator-error')).toHaveCount(0)
  })

  test('Consensus with no pairs shows the empty line', async ({ page }) => {
    await openWithOperatorReplies(page, { consensus: okEmpty({ consensus_pairs: [] }) })
    await page.getByTestId('operator-chip-consensus').click()
    await expect(page.getByTestId('operator-consensus-panel')).toBeVisible()
    await expect(page.getByTestId('operator-consensus-empty')).toContainText(
      'No corroboration pairs for topics in this hit set',
    )
    await expect(page.getByTestId('operator-consensus-list')).toHaveCount(0)
  })

  test('an operator call that fails with 500 surfaces operator-error; the hits stay', async ({
    page,
  }) => {
    await openWithOperatorReplies(page, {
      cluster: { status: 500, body: 'cluster operator blew up' },
    })
    await page.getByTestId('operator-chip-cluster').click()
    await expect(page.getByTestId('operator-error')).toHaveText('cluster operator blew up')
    await expect(page.getByTestId('search-workspace').locator('article')).toHaveCount(2)
  })
})
