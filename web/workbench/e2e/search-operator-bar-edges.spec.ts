import { expect, test, type Page, type Route } from '@playwright/test'
import { mainViewsNav, mockSignIn, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

/**
 * Search v3 §S4 operator bar — the edge state `search-operator-bar.spec.ts` (live) cannot reach: a
 * hit set with nothing graph-resolvable. It is a response shape, so `/api/search` is mocked.
 */

const QUERY = 'operator edges'

/** Hits with no episode_id / source_id — nothing the On-graph chip could focus. */
const ID_LESS_HITS = [
  { doc_id: 'transcript:1', score: 0.7, text: 'Hit one', metadata: { doc_type: 'transcript' } },
  { doc_id: 'transcript:2', score: 0.6, text: 'Hit two', metadata: { doc_type: 'transcript' } },
]

async function openWithIdLessHits(page: Page): Promise<void> {
  await mockSignIn(page, 'creator')
  await page.route('**/api/health**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'ok', corpus_library_api: true, search_api: true }),
    }),
  )
  await page.route('**/api/search?**', (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ query: QUERY, results: ID_LESS_HITS, error: null, detail: null }),
    }),
  )
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill('/mock/corpus')
  await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
  await page.locator('#search-q').fill(QUERY)
  await page.locator('#search-q').press('Enter')
  await expect(page.getByTestId('result-set-operator-bar')).toBeVisible()
}

test.describe('Search operator bar — edge states (mocked)', () => {
  test('a hit set with no graph ids disables "On graph (no ids)"', async ({ page }) => {
    await openWithIdLessHits(page)
    const chip = page.getByTestId('operator-chip-graph')
    await expect(chip).toHaveText('On graph (no ids)')
    await expect(chip).toBeDisabled()
  })
})
