import { test } from '@playwright/test'
import { sinkFor } from './sink'

/**
 * Wait for in-flight beacons to actually ARRIVE before the browser context is torn down.
 *
 * Umami's tracker sends with `fetch(url, { keepalive: true })` — measured in the served `script.js`.
 * `keepalive` survives a page UNLOAD, which is what it exists for, but it does not survive Playwright
 * closing the browser context at the end of a test: that tears down the network stack and cancels
 * whatever is in flight.
 *
 * The effect was intermittent and easy to misread. Specs passed — the recorder sees a request the
 * moment it STARTS — while `capture_created` was stored by two runs and missing from a third, and
 * `offline_session`, `empty_state_shown` and `highlights_export` were absent from Umami entirely.
 * Asserting only on the wire would have called all of that proven; asserting only on the surface would
 * have called it a missing event. It was neither: the events were correct and the test was hanging up
 * mid-sentence.
 *
 * This waits for the RESPONSES rather than sleeping a fixed interval, which is both exact and cheaper:
 * a fixed sleep is either too short sometimes or wasted every time.
 *
 * Importing this module registers the wait for the importing spec file.
 */
test.afterEach(async ({ page }) => {
  await sinkFor(page)?.settle()
})
