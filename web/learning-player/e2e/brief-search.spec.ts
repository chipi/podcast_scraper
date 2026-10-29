import { expect, test } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * The Brief's search, for what a listener remembers (operator 2026-10-10). Real backend, committed fixture corpus: a few words
 * from the episode find the sentence that says them, the words marked, playable from where they
 * were said, and saved as a highlight from there.
 */
test('a few remembered words find their sentence: marked, playable, and saved as a highlight', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'brief-search', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('The Bessent Tape').first().click()
  await page.getByTestId('player-open-insights').click()

  await page.locator('#kp-ask').fill('same point about credibility')
  await page.locator('#kp-ask').press('Enter')

  const transcript = page.getByTestId('kp-search-transcript')
  await expect(transcript).toBeVisible()
  const first = transcript.getByTestId('kp-search-piece').first()
  // The sentence that says it, not a ~300-word block — the remembered words marked.
  await expect(first).toContainText('the same point about credibility')
  await expect(first.locator('mark')).toContainText(['same', 'point', 'credibility'])
  // Timed from where it was said (the index stores no time; the server finds it).
  await expect(first.getByRole('button', { name: /Play from/ })).toBeVisible()

  await first.getByTestId('kp-search-highlight').click()
  await expect(page.getByText('Highlight saved')).toBeAttached()
})

test('a word search: every result holds a typed word, and each group folds', async ({ page }, testInfo) => {
  await signInIsolated(page, 'brief-search-words', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('The Bessent Tape').first().click()
  await page.getByTestId('player-open-insights').click()

  await page.locator('#kp-ask').fill('credibility')
  await page.locator('#kp-ask').press('Enter')
  const results = page.getByTestId('kp-search-results')
  await expect(results).toBeVisible()
  // Words, not meaning (operator 2026-10-12): a result found by meaning alone is not shown, so every
  // transcript piece and every insight has the word marked in it.
  const pieces = results.getByTestId('kp-search-piece')
  await expect(pieces.first()).toBeVisible()
  for (const p of await pieces.all()) await expect(p.locator('mark').first()).toBeVisible()
  for (const t of await results.getByTestId('kp-search-insights').locator('li').all()) {
    await expect(t).toContainText(/credib/i)
  }

  // One "Search results" section folds (operator 2026-10-12), its count in the header; inside it
  // the two groups are headings, always shown.
  await expect(results.getByTestId('kp-search-transcript')).toContainText('In the conversation')
  await expect(page.getByTestId('kp-section-toggle-search-transcript')).toHaveCount(0)
  const toggle = page.getByTestId('kp-section-toggle-search-results')
  await expect(toggle).toContainText(/Search results · \d+/)
  await toggle.click()
  await expect(pieces.first()).toBeHidden()
  await toggle.click()
  await expect(pieces.first()).toBeVisible()
})

// Operator 2026-10-15: an insight result carries what the Insights section shows — its type (mark and
// word), the asked word marked, and the bookmark that saves the same highlight.
test('an insight result shows its type, marks the asked word, and saves as a highlight', async ({ page }, testInfo) => {
  await signInIsolated(page, 'brief-search-insight', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('The Bessent Tape').first().click()
  await page.getByTestId('player-open-insights').click()
  await page.locator('#kp-ask').fill('reliability')
  await page.locator('#kp-ask').press('Enter')

  const insight = page.getByTestId('kp-search-insights').locator('li').first()
  await expect(insight.getByTestId('kp-search-insight-type')).toHaveText(/recommendation/i)
  await expect(insight.getByTestId('kp-search-insight-text').locator('mark')).toHaveText(/reliability/i)
  const save = insight.getByTestId('kp-search-insight-highlight')
  await save.click()
  await expect(save).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByText('Saved to highlights — tap to remove').first()).toBeAttached()
})
