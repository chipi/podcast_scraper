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

test('Your Week is absent when signed out (RFC-120: anon → /welcome, no digest)', async ({
  page,
}) => {
  // RFC-120: logged-out visitors land on /welcome, not HomeView. Your Week is per-user; it is
  // never rendered on the landing — so the invariant ("absent for anon") holds unchanged, but the
  // proof path is now the landing page, not a signed-out home.
  await page.goto('/')
  await expect(page).toHaveURL(/\/welcome/)
  await expect(page.getByText('Understand any podcast in minutes.')).toBeVisible() // landing rendered
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

  // ONE LINE, not four rows (#1978). #1591's contract is what this test is named for and it is
  // intact — the section still teaches instead of hiding. The four-row list was the implementation:
  // measured on a fresh account it stood 373px tall with zero episode links, sitting between the
  // hero and "What's new" and saying "… will land here" four times. Compacting it moved What's new
  // from y=771 to y=499 — above the fold on the surface every first-time tester lands on.
  const firstRun = yourWeek.getByTestId('yourweek-firstrun')
  await expect(firstRun.locator('li')).toHaveCount(0)
  await expect(firstRun).toContainText(/fills as you follow/i)
  // The one action that actually starts the digest survives; it is the whole point of teaching.
  await expect(firstRun.getByRole('link')).toHaveCount(1)

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
