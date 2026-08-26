import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Your Week — the in-app personal digest section on Home. REAL API over the committed validation
 * corpus (tests/fixtures/app-validation-corpus/v3), NO mocks.
 *
 * Coverage:
 *  - signed-out → absent entirely (the digest is per-user; there is nothing to teach an anonymous
 *    visitor). Signed-in with nothing due → a FIRST-RUN state, not a hidden section (#1591);
 *  - populated render: seed real per-user state via the REAL API (follow a show — the same
 *    add_subscription the tier-3 backend test seeds), then the "new in your follows" rollup renders
 *    deterministically (no date/heard/spaced-repetition dependence — it only needs an unheard,
 *    graph-carrying episode, which every corpus episode has).
 */

test('Your Week is absent when signed out', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText("Find any moment you've heard.")).toBeVisible() // home rendered
  await expect(page.getByTestId('your-week')).toHaveCount(0)
})

test('Your Week teaches a fresh signed-in user instead of hiding (#1591)', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'your-week-empty', testInfo) // asserts signed-in (Sign out visible)
  await page.goto('/')

  // REVERSED contract. This previously asserted the section must NOT render when nothing is due.
  // Hiding meant the user most in need of learning that a weekly digest exists — a brand-new one —
  // got no hint of it, and an API outage was indistinguishable from a quiet week. See UXS-012.
  const yourWeek = page.getByTestId('your-week')
  await expect(yourWeek).toBeVisible()
  await expect(yourWeek.getByTestId('yourweek-firstrun')).toBeVisible()

  // One row per digest section, each saying what will appear there and how to earn it. #1836 added
  // the "topics & people you follow" section, so there are four teaching rows now (was three).
  await expect(yourWeek.getByTestId('yourweek-firstrun').locator('li')).toHaveCount(4)
  await expect(yourWeek.getByText('New in your follows')).toBeVisible()
  await expect(yourWeek.getByText('New in topics & people you follow')).toBeVisible()

  // Nothing to expand yet, so no compact/full toggle.
  await expect(yourWeek.getByTestId('yourweek-toggle')).toHaveCount(0)
})

test('Your Week renders the follows rollup after the user follows a show', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'your-week-follows', testInfo)

  // Seed via the REAL API (the shape the tier-3 backend test seeds): follow a show that carries a
  // graph — its unheard episodes become the "new in your follows" section.
  const resp = await page.request.get('/api/app/episodes?page_size=50')
  expect(resp.ok()).toBeTruthy()
  const items = (await resp.json()).items as Array<{ feed_id: string; has_kg?: boolean }>
  const seed = items.find((e) => e.has_kg) ?? items[0]
  expect(seed?.feed_id).toBeTruthy()
  const follow = await page.request.post('/api/app/library', { data: { feed_id: seed.feed_id } })
  expect(follow.ok()).toBeTruthy()

  await page.goto('/')
  const yourWeek = page.getByTestId('your-week')
  await expect(yourWeek).toBeVisible()
  await expect(yourWeek.getByRole('link').first()).toBeVisible() // at least one highlight card

  // Expand to the full layout and confirm it is the follows section that surfaced.
  await yourWeek.getByTestId('yourweek-toggle').click()
  await expect(yourWeek.getByText('New in your follows')).toBeVisible()
})
