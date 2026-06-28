import { expect, test } from '@playwright/test'

/**
 * P3 Consolidation end-to-end — REAL API over the COMMITTED validation corpus (now carrying RFC-088
 * enrichment envelopes), NO mocks. Covers the consumer enrichment read surface (#1121), the Recall
 * scope toggle (#1124), the "your corpus" entity lens (#1125), and the Revisit inbox (#1125).
 * Per-user state is the gitignored APP_DATA_DIR.
 */
test('enrichment read surface + recall toggle + your-corpus lens + Revisit inbox', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('link', { name: 'Sign in' }).click()
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()

  // Open an episode and capture its slug from the URL.
  await page.goto('/')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page.getByText(/Index funds are not a strategy/).first()).toBeVisible()
  const slug = new URL(page.url()).pathname.split('/').pop()!

  // #1121: the consumer enrichment read surface serves the committed envelopes.
  const epEnrich = await page.request.get(`/api/app/episodes/${slug}/enrichment`)
  expect(epEnrich.ok()).toBeTruthy()
  expect((await epEnrich.json()).signals).toHaveProperty('topic_cooccurrence')
  const corpusEnrich = await page.request.get('/api/app/corpus/enrichment')
  expect((await corpusEnrich.json()).signals).toHaveProperty('temporal_velocity')

  // #1125: the entity-card "your corpus" lens — open a topic chip from the Insights panel, then
  // toggle to My corpus (the card refetches scoped to the heard set; it stays rendered).
  await page.getByRole('button', { name: 'Insights' }).first().click()
  await page.locator('button.text-topic, button.text-person').first().click()
  const cardScope = page.getByRole('tablist', { name: 'Card scope' })
  await expect(cardScope).toBeVisible()
  await cardScope.getByRole('tab', { name: 'My corpus' }).click()
  await expect(cardScope.getByRole('tab', { name: 'My corpus' })).toHaveAttribute(
    'aria-selected',
    'true',
  )

  // #1124: Recall — the search scope toggle switches to a corpus-scoped grounded search. (The
  // mock user's corpus content varies across the shared e2e run, so assert the toggle is functional
  // rather than a specific result set — recall semantics are unit-tested.)
  await page.goto('/search?q=index')
  const searchScope = page.getByRole('tablist', { name: 'Search scope' })
  await expect(searchScope).toBeVisible()
  await searchScope.getByRole('tab', { name: 'My corpus' }).click()
  await expect(searchScope.getByRole('tab', { name: 'My corpus' })).toHaveAttribute(
    'aria-selected',
    'true',
  )

  // #1125: the Revisit inbox + pacing control. (Pacing state is shared across the parallel e2e
  // projects, so assert the toggle flips its label rather than a specific starting state.)
  await page.goto('/library')
  await page.getByRole('button', { name: 'Revisit' }).click()
  const pacing = page.getByRole('button', { name: /^(Pause|Resume)$/ })
  await expect(pacing).toBeVisible()
  const before = await pacing.textContent()
  await pacing.click()
  await expect(pacing).not.toHaveText(before ?? '')
})
