import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * P3 Consolidation end-to-end — REAL API over the COMMITTED validation corpus (now carrying RFC-088
 * enrichment envelopes), NO mocks. Covers the consumer enrichment read surface (#1121), the Recall
 * scope toggle (#1124), the "your corpus" entity lens (#1125), and the Revisit inbox (#1125).
 * Per-user state is the gitignored APP_DATA_DIR.
 */
test('enrichment read surface + recall toggle + your-corpus lens + Revisit inbox', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'consolidation', testInfo)

  // Open an episode and capture its slug from the URL.
  await page.goto('/')
  await page.goto('/podcast/p09') // #1148: open a RECENT episode so it isn't due for resurfacing
  await page.getByText('Risk Is a Systems Property').first().click()
  await expect(page.getByText(/Risk lives in the couplings/).first()).toBeVisible()
  const slug = new URL(page.url()).pathname.split('/').pop()!

  // #1121: the consumer enrichment read surface serves the committed envelopes.
  const epEnrich = await page.request.get(`/api/app/episodes/${slug}/enrichment`)
  expect(epEnrich.ok()).toBeTruthy()
  expect((await epEnrich.json()).signals).toHaveProperty('insight_density')
  const corpusEnrich = await page.request.get('/api/app/corpus/enrichment')
  expect((await corpusEnrich.json()).signals).toHaveProperty('temporal_velocity')

  // #1125: the entity-card "your corpus" lens — open a topic chip from the Insights panel, then
  // toggle to My corpus (the card refetches scoped to the heard set; it stays rendered).
  await page.getByRole('button', { name: 'Insights' }).first().click()
  // insight_density strip renders at the head of the Insights list (Plan B #2).
  await expect(page.getByTestId('episode-density')).toBeVisible()
  await page.locator('button.text-topic, button.text-person').first().click()
  const cardScope = page.getByRole('tablist', { name: 'Card scope' })
  await expect(cardScope).toBeVisible()
  await cardScope.getByRole('tab', { name: 'My corpus' }).click()
  await expect(cardScope.getByRole('tab', { name: 'My corpus' })).toHaveAttribute(
    'aria-selected',
    'true',
  )

  // #1124: Recall — switch the scope to "My corpus". This isolated user has captured nothing, so the
  // scoped search is honest-empty. The corpus vector index builds lazily + slowly (> the per-test
  // budget) on first search, so a cold index surfaces "Search needs the library index" instead;
  // either is a valid no-results state for a fresh user, and accepting both keeps this off the
  // first-search-races-the-build flake without weakening the scope-toggle assertion.
  await page.goto('/search?q=index')
  const searchScope = page.getByRole('tablist', { name: 'Search scope' })
  await expect(searchScope).toBeVisible()
  await searchScope.getByRole('tab', { name: 'My corpus' }).click()
  await expect(
    page.getByText(/Nothing in your corpus on this yet|Search needs the library index/),
  ).toBeVisible()

  // #1125: the Revisit inbox — a fresh user has nothing due; the pacing control pauses.
  await page.goto('/library')
  await page.getByRole('button', { name: 'Revisit' }).click()
  await expect(page.getByText(/Nothing to revisit right now/)).toBeVisible()
  await page.getByRole('button', { name: 'Pause' }).click()
  await expect(page.getByText('Resurfacing is paused.')).toBeVisible()
})
