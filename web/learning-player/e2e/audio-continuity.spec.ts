import { expect, test } from '@playwright/test'
import { navTo } from './helpers'

/**
 * Audio survives navigation (#1587) — the property the whole change exists for.
 *
 * The `<audio>` element used to be rendered by PlayerView, so leaving an episode unmounted it and
 * playback stopped. You could not browse, search or look anything up while listening — and the
 * learning features are most valuable mid-listen, so every one of those journeys cost you the thing
 * you were listening to. This asserts the fix at the only level that proves it: a real browser,
 * across a real route change.
 */

test('playback continues across navigation, and the mini-player offers the way back', async ({
  page,
}) => {

  // Reach the episode via its show page — date-independent, same route the other specs use.
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  // Start playback and confirm the element is actually advancing, not merely "not erroring".
  await page.getByRole('button', { name: 'Play', exact: true }).first().click()
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.currentTime ?? 0), {
      timeout: 15_000,
    })
    .toBeGreaterThan(0.2)

  const episodeUrl = page.url()

  // No mini-player on the player page itself — the full transport is right there.
  await expect(page.getByTestId('mini-player')).toHaveCount(0)

  // Navigate away IN-APP. Must be a router navigation, not page.goto — a full reload tears down
  // the whole SPA and no amount of store ownership survives that. Client-side navigation is the
  // case that was broken: the view unmounted and took the <audio> element with it.
  // Whichever nav this viewport actually shows — the bottom bar is mobile-only and the header icon
  // links are desktop-only, so either hard-coded choice runs on one project and fails on the other.
  await navTo(page, 'search')
  await expect(page).toHaveURL(/\/search/)
  await expect(page.getByTestId('mini-player')).toBeVisible()

  const atNav = await page.evaluate(() => document.querySelector('audio')?.currentTime ?? 0)
  expect(atNav, 'audio element must still exist after leaving the player').toBeGreaterThan(0)

  // Still ADVANCING, not just present — a paused-but-alive element would be a different bug.
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.currentTime ?? 0), {
      timeout: 15_000,
    })
    .toBeGreaterThan(atNav)

  // The mini-player is the way back to what is playing.
  await page.getByTestId('mini-player-open').click()
  await expect(page).toHaveURL(episodeUrl)
  await expect(page.getByTestId('mini-player')).toHaveCount(0)
})

test('the mini-player pauses and resumes from anywhere', async ({ page }) => {

  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await page.getByRole('button', { name: 'Play', exact: true }).first().click()
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.currentTime ?? 0), {
      timeout: 15_000,
    })
    .toBeGreaterThan(0.2)

  await navTo(page, 'catalog')
  const toggle = page.getByTestId('mini-player-toggle')
  await expect(toggle).toBeVisible()

  await toggle.click()
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.paused ?? true))
    .toBe(true)

  await toggle.click()
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.paused ?? true))
    .toBe(false)
})

/**
 * The bars must not eat page content (#1594 + #1587).
 *
 * Both are `position: fixed`, so they cover the end of every page unless `main` reserves room. It
 * shipped reserving a CONSTANT 96px on mobile and 24px on desktop, against a tab bar plus a
 * mini-player — so the last card of any list sat underneath the transport whenever something was
 * playing. The unit check asserted the padding CLASSES existed, which is exactly why nobody saw it.
 *
 * Clickability is the assertion, because that is the actual harm: Playwright's click fails if
 * another element intercepts the pointer, which is precisely what an overlapping fixed bar does.
 */
test('the last item on a page stays reachable while audio is playing', async ({ page }) => {

  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await page.getByRole('button', { name: 'Play', exact: true }).first().click()
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.currentTime ?? 0), {
      timeout: 15_000,
    })
    .toBeGreaterThan(0.2)

  // Somewhere with a long list, and the mini-player up.
  await navTo(page, 'catalog')
  await expect(page.getByTestId('mini-player')).toBeVisible()

  // Scoped to <main>: the mini-player itself contains an /episode/ link (it is the way back), so an
  // unscoped selector measures the bar against itself and reports a 51px overlap that is really the
  // bar's own height. Cost me a round of chasing a fix for a bug the selector invented.
  const cards = page.locator('main a[href^="/episode/"]')
  await expect(cards.first()).toBeVisible()
  const last = cards.last()

  // Wait for the list to actually render before scrolling — scrolling an unpopulated page is a
  // no-op, which silently turns this whole assertion into a tautology.
  await expect.poll(async () => cards.count()).toBeGreaterThan(5)
  await expect
    .poll(async () =>
      page.evaluate(() => document.documentElement.scrollHeight - window.innerHeight),
    )
    .toBeGreaterThan(200)

  // Scroll to the true bottom: the reservation only has to hold at the end of the document, which
  // is exactly where a too-small padding stops being enough. Repeat until it stops moving — the
  // page keeps growing as artwork loads, so a single scrollTo lands short of the real bottom and
  // every measurement below it is off by that remainder.
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const before = window.scrollY
          window.scrollTo(0, document.documentElement.scrollHeight)
          return window.scrollY - before
        }),
      { timeout: 15_000 },
    )
    .toBe(0)
  expect(await page.evaluate(() => window.scrollY), 'the page must have scrolled').toBeGreaterThan(0)

  // Geometry, not a trial click: a trial click only probes the element's CENTRE, so a bar covering
  // the bottom half of the last card passes it. (Verified — the trial-click version of this test
  // passed against the known-broken pb-24 padding.) Compare edges instead.
  const barBox = await page.getByTestId('mini-player').boundingBox()
  expect(barBox, 'mini-player must be on screen for this assertion to mean anything').not.toBeNull()
  await expect(last).toBeVisible()

  // Measure main's CONTENT box, not the last card. The last card is not flush against the end of
  // the content — there is spacing after it — so a card-based assertion passes with a padding value
  // that genuinely occludes content. (Verified: the card version passed against the known-broken
  // pb-24.) Where the reserved space ends is the property that actually has to hold.
  const contentBottom = await page.evaluate(() => {
    const m = document.querySelector('main') as HTMLElement
    const rect = m.getBoundingClientRect()
    return rect.bottom - parseFloat(getComputedStyle(m).paddingBottom)
  })

  expect(
    contentBottom,
    'main must reserve enough bottom padding that its content ends above the mini-player',
  ).toBeLessThanOrEqual(barBox!.y)
})
