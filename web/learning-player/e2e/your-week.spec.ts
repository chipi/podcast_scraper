import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Your Week — the in-app personal digest section on Home. REAL API over the committed validation
 * corpus (tests/fixtures/app-validation-corpus/v3), NO mocks.
 *
 * Coverage:
 *  - hidden contract: signed-out, and a fresh signed-in user with no activity → the section must
 *    not render (no empty shell, no error);
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

test('Your Week stays hidden for a fresh signed-in user with no activity', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'your-week-empty', testInfo) // asserts signed-in (Sign out visible)
  await page.goto('/')
  // No captures, heard episodes, or follows yet → nothing due → the section must not render.
  await expect(page.getByTestId('your-week')).toHaveCount(0)
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
