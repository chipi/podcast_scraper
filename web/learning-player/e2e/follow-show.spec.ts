import { expect, test, type APIRequestContext } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Follow a show from the UI — the feed-subscription affordance that fills the "new in your follows"
 * section of Your Week. REAL API over the committed validation corpus, NO mocks.
 *
 * The sibling your-week.spec.ts seeds the subscription by POSTing /api/app/library directly; this
 * spec covers the gap that motivated the button: the user can reach that same state from the UI.
 *
 * Coverage:
 *  - the button is per-user (absent signed-out);
 *  - clicking it lands the feed in GET /api/app/library and survives a reload;
 *  - after following a graph-carrying show, Your Week appears on Home with the follows rollup;
 *  - clicking again unfollows (the library goes back to empty).
 */

/** A graph-carrying episode from the corpus — its show yields "new in your follows" content. */
async function seedFeedId(request: APIRequestContext): Promise<string> {
  const resp = await request.get('/api/app/episodes?page_size=50')
  expect(resp.ok()).toBeTruthy()
  const items = ((await resp.json()) as { items: Array<{ feed_id: string; has_kg?: boolean }> }).items
  const seed = items.find((e) => e.has_kg) ?? items[0]
  expect(seed?.feed_id).toBeTruthy()
  return seed.feed_id
}

test('signed out, the follow button is a sign-in teaser rather than absent (#1590)', async ({
  page,
}) => {
  // This asserted the button was ABSENT. The show page is the primary follow surface, so hiding it
  // there hid the capability from every visitor deciding whether an account is worth making —
  // the defect #1590 exists to fix. It renders; the tap defers to sign-in and comes back here.
  const feedId = await seedFeedId(page.request)
  await page.goto(`/podcast/${encodeURIComponent(feedId)}`)
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible() // show page rendered

  const follow = page.getByTestId('follow-show')
  await expect(follow).toBeVisible()
  await expect(follow).toHaveAttribute('aria-label', 'Sign in to follow')
  // Nothing is toggled, so claiming a pressed state would be a lie to assistive tech.
  await expect(follow).not.toHaveAttribute('aria-pressed', /.*/)

  await follow.click()
  await expect(page).toHaveURL(new RegExp(`/login\\?redirect=.*${encodeURIComponent(feedId)}`))

  // And the library really is untouched — no optimistic write leaked past the gate.
  const lib = await page.request.get('/api/app/library')
  expect(lib.status()).toBe(401)
})

test('following a show from the show page lands in the library and surfaces Your Week', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'follow-show', testInfo)
  const feedId = await seedFeedId(page.request)

  await page.goto(`/podcast/${encodeURIComponent(feedId)}`)
  const follow = page.getByTestId('follow-show')
  await expect(follow).toBeVisible()
  await expect(follow).toHaveAttribute('aria-pressed', 'false')

  await follow.click()
  await expect(follow).toHaveAttribute('aria-pressed', 'true')
  await expect(follow).toContainText('Following')

  // The server took it — not just an optimistic flip.
  await expect
    .poll(async () => {
      const lib = await page.request.get('/api/app/library')
      const items = (await lib.json() as { items: Array<{ feed_id: string }> }).items
      return items.map((i) => i.feed_id)
    })
    .toContain(feedId)

  // Follow-state is loaded from the server, not just local component state.
  await page.reload()
  await expect(page.getByTestId('follow-show')).toContainText('Following')

  // The acceptance criterion: the digest section now has content to show.
  await page.goto('/')
  const yourWeek = page.getByTestId('your-week')
  await expect(yourWeek).toBeVisible()
  await yourWeek.getByTestId('yourweek-toggle').click()
  await expect(yourWeek.getByText('New in your follows')).toBeVisible()
})

test('clicking Following unfollows the show', async ({ page }, testInfo) => {
  await signInIsolated(page, 'unfollow-show', testInfo)
  const feedId = await seedFeedId(page.request)

  await page.goto(`/podcast/${encodeURIComponent(feedId)}`)
  const follow = page.getByTestId('follow-show')
  await follow.click()
  await expect(follow).toContainText('Following')

  await follow.click()
  await expect(follow).toContainText('Follow show')
  await expect
    .poll(async () => {
      const lib = await page.request.get('/api/app/library')
      return (await lib.json() as { items: unknown[] }).items.length
    })
    .toBe(0)
})
