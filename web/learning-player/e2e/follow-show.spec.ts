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

test('signed out, a show deep-link redirects to the landing with a ?redirect funnel (#1590, RFC-120)', async ({
  page,
}) => {
  // RFC-120: every non-public route redirects a logged-out visitor to /welcome with ?redirect back
  // to the intended path. This is the new "teaser" UX that replaced the per-button sign-in hint:
  // instead of rendering the show page with a gated follow button, the visitor sees the full lure
  // landing — which is a richer sign-up pitch — and is sent back to the show after signing in.
  //
  // Use a static corpus feedId (p05, committed corpus). The /api/app/episodes endpoint that the
  // other tests seed from is itself auth-gated, so a signed-out test cannot call it.
  const feedId = 'p05'
  await page.goto(`/podcast/${encodeURIComponent(feedId)}`)

  // Must land on /welcome, not on the show page.
  await expect(page).toHaveURL(/\/welcome/)
  // The redirect param threads the intended destination through signup.
  await expect(page).toHaveURL(new RegExp(`redirect=.*${encodeURIComponent(feedId)}`))
  // The landing CTA is the primary action, pointing to signup with the redirect preserved.
  await expect(page.getByTestId('landing-cta-primary')).toBeVisible()

  // And the library API confirms the session is unauthenticated.
  const lib = await page.request.get('/api/app/library')
  expect(lib.status()).toBe(401)
})

test("following a show from the show page lands in the library and makes What's new yours", async ({
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
  await expect(follow).toHaveText('Following')

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

  // The acceptance criterion: Home's What's new is now from the followed show (operator 2026-10-07 —
  // it was Your Week's "New in your follows", which What's new took over).
  await page.goto('/')
  await expect(page.getByText('New in your shows and topics', { exact: false })).toBeVisible()
})

test('clicking Following unfollows the show', async ({ page }, testInfo) => {
  await signInIsolated(page, 'unfollow-show', testInfo)
  const feedId = await seedFeedId(page.request)

  await page.goto(`/podcast/${encodeURIComponent(feedId)}`)
  const follow = page.getByTestId('follow-show')
  await follow.click()
  await expect(follow).toHaveText('Following')

  await follow.click()
  await expect(follow).toHaveText('Follow')
  await expect
    .poll(async () => {
      const lib = await page.request.get('/api/app/library')
      return (await lib.json() as { items: unknown[] }).items.length
    })
    .toBe(0)
})

test('the show page surfaces feed authors and last-updated (#2043)', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'feed-meta', testInfo)
  // p05 (Long Horizon Notes) — the fixture feed carries author "Nora Bakker".
  await page.goto('/podcast/p05')

  // The by-line is no longer its own element: authors are the first entry of the single composed
  // meta line (`metaLine` in PodcastView), so the separators fall between the fields that are
  // actually present. That line renders TWICE with one visible at a time — `podcast-feed-meta`
  // under the artwork on a phone, `podcast-feed-meta-wide` under the title from `sm` up — because
  // ~85 characters cannot sit in a 144px gutter. This spec runs on both the mobile and desktop
  // projects, so it matches whichever copy is actually visible rather than picking one and failing
  // on the other.
  const meta = page.locator(
    '[data-testid="podcast-feed-meta"]:visible, [data-testid="podcast-feed-meta-wide"]:visible',
  )
  await expect(meta).toContainText('Nora Bakker')
  await expect(meta).toContainText('Updated')

  // LANGUAGE is back (V2-C.1): the corpus is multilingual now, so the chip says something. It is a
  // LanguageBadge leading the line, carrying the normalized code — never the raw feed tag `en-us`.
  await expect(meta.getByTestId('language-badge')).toHaveAttribute('data-lang', 'en')
  await expect(meta).not.toContainText('en-us')
})
