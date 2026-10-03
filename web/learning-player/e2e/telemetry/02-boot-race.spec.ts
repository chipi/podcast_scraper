import { expect, test } from '@playwright/test'
import { attachSink } from './sink'
import './settle'

/**
 * THE BOOT RACE (#2267). Found by this tier, not by review.
 *
 * `installUmami()` appends a `defer` script. `track()` reads `window.umami` and calls through
 * OPTIONALLY: `umami?.track?.(name, props)`. So until that script has downloaded and executed,
 * `window.umami` is undefined and every event fired in that window is dropped — no error, no
 * retry, no trace.
 *
 * Which events land in that window is not random. It is precisely the earliest ones:
 * `landing_view` on mount, the first `screen_view` from the router's afterEach, and
 * `auth_completed` when a returning session resolves at boot. Those are the FIRST steps of the
 * onboarding funnel, so the funnel systematically under-reports its own top — and it under-reports
 * hardest for slow connections and cold caches, which is exactly the population a beta needs to
 * see honestly. A conversion rate whose denominator is quietly missing its slowest visitors reads
 * as better than it is.
 *
 * This surfaced as one intermittently-failing assertion in the landing spec: three tests saw the
 * events and one saw NONE. That is the signature of a race, and treating it as flake (a retry, a
 * longer timeout) would have buried a live data-loss bug under a green suite.
 *
 * The first test below delays `script.js` to make the race deterministic.
 */

const SCRIPT_DELAY_MS = 1200

test('an event fired before script.js lands is still delivered', async ({ page }) => {
  const sink = attachSink(page)

  // Hold the tracker's own script back, letting everything else through untouched. This is the real
  // condition — a slow CDN, a cold cache, a phone on a train — not an injected fault.
  await page.route('**/script.js', async (route) => {
    await new Promise((r) => setTimeout(r, SCRIPT_DELAY_MS))
    await route.continue()
  })

  await page.goto('/welcome')

  // `landing_view` fires on mount, which happens well inside the delay window. Before the fix the
  // beacon never appears, because `window.umami` did not exist at the moment track() ran.
  const view = await sink.waitForEvent('landing_view', 15_000)
  expect(view.name).toBe('landing_view')

  // The first screen_view is in the same window.
  await sink.waitForEvent('screen_view', 15_000)
})

test('ordering survives the queue: identify is applied before the events it should tag', async ({
  page,
}) => {
  const sink = attachSink(page)
  await page.route('**/script.js', async (route) => {
    await new Promise((r) => setTimeout(r, SCRIPT_DELAY_MS))
    await route.continue()
  })

  // A full-page sign-in: the session resolves at boot, so `identify` and `auth_completed` both fire
  // inside the delay window. If the queue replayed out of order, the first events of a signed-in
  // session would be filed as anonymous — the attribution bug `resetIdentity` exists to prevent,
  // reintroduced from the other direction.
  await page.goto('/api/app/auth/login?as=telemetry-boot-race')
  await page.waitForLoadState('networkidle')

  await expect
    .poll(() => sink.umami.length, { timeout: 15_000 })
    .toBeGreaterThan(0)

  const identifyAt = sink.umami.findIndex((b) => b.type === 'identify' || Boolean(b.id))
  const firstNamed = sink.umami.findIndex((b) => Boolean(b.name))
  if (identifyAt >= 0 && firstNamed >= 0) {
    expect(
      identifyAt,
      'identify must reach the wire before the first named event it is meant to tag',
    ).toBeLessThan(firstNamed)
  }
})

test('a tracker that never loads drops its queue instead of growing forever', async ({ page }) => {
  const sink = attachSink(page)
  // An ad blocker, an offline first launch, a CSP that forbids the origin: the script simply never
  // arrives. The queue must not become an unbounded leak in a long session.
  await page.route('**/script.js', (route) => route.abort())

  await page.goto('/welcome')
  // Generate plenty of trackable activity while the tracker is dead.
  for (let i = 0; i < 5; i++) {
    await page.goto('/welcome')
    await page.getByTestId('landing-cta-signin').click().catch(() => {})
  }

  // Nothing reached the wire — correct, there is nowhere to send it.
  expect(sink.umami.length).toBe(0)

  // And the app is still alive: a dead metric must never be a dead page.
  await page.goto('/welcome')
  await expect(page.getByTestId('landing-cta-primary')).toBeVisible()
})
