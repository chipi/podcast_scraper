import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * The reach chip must not render as an empty pill when reach is withheld (#1957).
 *
 * `GET /api/app/episodes/{slug}/stats` nulls `listeners` and `opens` below the k-anonymity floor
 * of 5 distinct listeners (#1923), and returns `daily: []` with them. Every child of the reach
 * chip is gated on that data — but its wrapper was gated only on `!panelOpen`, so it painted a
 * rounded background, padding and a backdrop blur around nothing.
 *
 * The fixture corpus has no episode above the floor, which is the same state production is in
 * (measured: 12 of 12 sampled episodes withheld). So this spec reproduces the real condition
 * rather than a contrived one, and the assertion is simply that nothing is drawn.
 */
test('no empty reach pill when the k-anonymity floor withholds the numbers', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'reach-pill', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText(/Index Investing Without the Myths/).first().click()
  await expect(page).toHaveURL(/\/episode\//)

  // The insights control shares the row with the reach chip — waiting for it means the row has
  // rendered, so a missing reach chip is a real absence and not a race.
  await expect(page.getByTestId('player-open-insights')).toBeVisible()

  const stats = await page.evaluate(async () => {
    const slug = window.location.pathname.split('/').pop()
    const r = await fetch(`/api/app/episodes/${slug}/stats`, { credentials: 'include' })
    return (await r.json()) as { listeners: number | null; opens: number | null }
  })

  // Precondition: this only tests what it claims while the numbers really are withheld. If the
  // fixture ever gains an episode above the floor, this must be re-pointed rather than silently
  // asserting nothing.
  expect(
    stats.listeners,
    'fixture episode is expected to be below the k-anonymity floor',
  ).toBeNull()

  await expect(
    page.getByTestId('player-reach'),
    'withheld reach must render NOTHING — an empty rounded pill reads as a rendering fault',
  ).toHaveCount(0)
})
