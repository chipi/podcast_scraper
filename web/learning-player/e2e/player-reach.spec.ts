import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * The reach chip renders only when it has something to say (#1957).
 *
 * `GET /api/app/episodes/{slug}/stats` withholds `listeners` and `opens` below the k-anonymity
 * floor of 5 distinct listeners (#1923), returning `daily: []` with them. Every child of the reach
 * chip is gated on that data — but its wrapper was gated only on `!panelOpen`, so it painted a
 * rounded background, padding and a backdrop blur around nothing.
 *
 * ## Why this mocks the endpoint instead of using real data
 *
 * The first version asserted against whatever the fixture happened to hold, with a precondition
 * that the episode was below the floor. That passed alone and FAILED IN CI, because the suite
 * shares `APP_DATA_DIR`: every spec that opens an episode records another distinct listener, so by
 * the time this one ran the episode had crossed the floor of 5. The test's meaning depended on how
 * many other tests had run before it — which is not a property a test may have.
 *
 * Withholding is a CONTRACT, so it is stated as one here. Both sides are asserted, because a fix
 * that simply never renders the chip would also make a withheld-only test pass.
 */
const STATS = /\/api\/app\/episodes\/[^/]+\/stats/

async function openEpisode(page: import('@playwright/test').Page): Promise<void> {
  await page.goto('/podcast/p05')
  await page.getByText(/Index Investing Without the Myths/).first().click()
  await expect(page).toHaveURL(/\/episode\//)
  // Shares the row with the reach chip: once this is up the row has rendered, so an absent chip
  // is a real absence rather than a race.
  await expect(page.getByTestId('player-open-insights')).toBeVisible()
}

test('withheld reach renders nothing, not an empty pill', async ({ page }, testInfo) => {
  await signInIsolated(page, 'reach-withheld', testInfo)
  await page.route(STATS, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      // Exactly what the server returns below the floor.
      body: JSON.stringify({ slug: 'x', listeners: null, opens: null, insights: 3, daily: [] }),
    }),
  )

  await openEpisode(page)

  await expect(
    page.getByTestId('player-reach'),
    'withheld reach must render NOTHING — an empty rounded pill reads as a rendering fault',
  ).toHaveCount(0)
})

test('disclosed reach still renders the chip', async ({ page }, testInfo) => {
  // The other half of the contract. Without this, "delete the chip entirely" would pass.
  await signInIsolated(page, 'reach-disclosed', testInfo)
  await page.route(STATS, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        slug: 'x',
        listeners: 42,
        opens: 96,
        insights: 3,
        daily: [{ date: '2026-07-20', count: 7 }],
      }),
    }),
  )

  await openEpisode(page)

  await expect(
    page.getByTestId('player-reach'),
    'reach above the floor must still be shown — the fix must hide the EMPTY chip, not the chip',
  ).toBeVisible()
})
