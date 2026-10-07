import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Your Week — your week in review on Home (operator 2026-10-07). REAL API over the committed
 * validation corpus (tests/fixtures/app-validation-corpus/v3), NO mocks.
 *
 * Coverage:
 *  - signed-out → absent entirely. Signed-in with nothing yet → a FIRST-RUN line (#1591);
 *  - populated render: play an episode through the REAL API (two position saves, so listening time
 *    accrues), then "You listened to" renders. New episodes from followed shows are NOT here any more
 *    — they are What's new — so a follow alone leaves Your Week on its first-run line.
 */

/** Play an episode the way the player does: two saves, two minutes apart in position. */
async function listenToOne(page: import('@playwright/test').Page): Promise<string> {
  const resp = await page.request.get('/api/app/episodes?page_size=1')
  const slug = ((await resp.json()).items as Array<{ slug: string }>)[0].slug
  const tz = -new Date().getTimezoneOffset()
  for (const position_seconds of [10, 130]) {
    const r = await page.request.put(`/api/app/playback/${slug}`, { data: { position_seconds, tz_offset_minutes: tz } })
    expect(r.ok()).toBeTruthy()
  }
  return slug
}

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

  // While the welcome card asks a brand-new listener for interests, the card IS the teaching and an
  // empty Your Week stays out of its way (operator 2026-10-07). Declining brings the first-run line.
  await expect(page.getByTestId('interests-welcome')).toBeVisible()
  await expect(page.getByTestId('your-week')).toHaveCount(0)
  await page.getByRole('button', { name: 'Not now' }).click()

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
  await expect(firstRun).toContainText(/fills in as you listen and save/i)
  // The one action that actually starts the digest survives; it is the whole point of teaching.
  await expect(firstRun.getByRole('link')).toHaveCount(1)

  // Nothing to expand yet, so no compact/full toggle.
  await expect(yourWeek.getByTestId('yourweek-toggle')).toHaveCount(0)
})

test('Your Week shows what you listened to this week, and not the new episodes of a followed show', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'your-week-listened', testInfo)
  // A follow alone: its new episodes are What's new's now, so Your Week stays on its first-run line.
  const resp = await page.request.get('/api/app/episodes?page_size=50')
  const items = (await resp.json()).items as Array<{ feed_id: string }>
  expect((await page.request.post('/api/app/library', { data: { feed_id: items[0].feed_id } })).ok()).toBeTruthy()
  await listenToOne(page)

  await page.goto('/')
  const yourWeek = page.getByTestId('your-week')
  await expect(yourWeek).toBeVisible()
  await expect(yourWeek.getByRole('link').first()).toBeVisible()
  await yourWeek.getByTestId('yourweek-toggle').click()
  await expect(yourWeek.getByText('You listened to')).toBeVisible()
  await expect(yourWeek.getByText('New in your follows')).toHaveCount(0)
})

/**
 * Your Week never shows the digest's `revisit` section (UXS-012, 2026-09-30).
 *
 * The same payload feeds the email digest, which keeps its revisit content; on Home that section
 * repeated what the RevisitRail already shows. A fresh corpus account has no highlight old enough to
 * resurface, so the real response carries no revisit section — the check would pass vacuously. The
 * real response is therefore fetched and a revisit section ADDED to it (the focused-mock exception
 * `long-show-title.spec.ts` makes), and the listened section beside it must still render.
 */
test('Your Week drops the revisit section even when the digest carries one', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'your-week-no-revisit', testInfo)
  await listenToOne(page)

  const REVISIT_TITLE = 'REVISIT ITEM THAT MUST NOT RENDER'
  await page.route('**/api/app/your-week', async (route) => {
    const real = await route.fetch()
    const body = await real.json()
    body.sections = [
      ...(body.sections ?? []),
      {
        kind: 'revisit',
        items: [{ episode_slug: 'x', episode_title: REVISIT_TITLE, deep_link: '/', quote: 'q', t_ms: 0 }],
      },
    ]
    await route.fulfill({ response: real, json: body })
  })

  await page.goto('/')
  const yourWeek = page.getByTestId('your-week')
  await expect(yourWeek).toBeVisible()
  await yourWeek.getByTestId('yourweek-toggle').click()
  await expect(yourWeek.getByText('You listened to')).toBeVisible()
  await expect(page.getByText(REVISIT_TITLE)).toHaveCount(0)
  // The handler fetches the real response, so a refetch still in flight when the test ends would
  // throw "while running route callback" and fail the whole run outside any test.
  await page.unrouteAll({ behavior: 'ignoreErrors' })
})
