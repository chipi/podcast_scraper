import { expect, test } from '@playwright/test'
import { attachSink } from './sink'
import './settle'

/**
 * One page view per navigation (prod 2026-10-04).
 *
 * A single prod visitor logged THREE page views per navigation. Two causes, both in
 * `services/analytics.ts`: every signed-out `refresh()` replaced the tracker (each fresh script sends
 * its own initial page view), and each replacement left the previous script's `pushState` wrapper
 * in place, so every later navigation was sent once per tracker ever installed. Unit tests prove the
 * mechanism; only the real script on the wire proves the count.
 */
test('an anonymous visitor sends exactly one page view per navigation', async ({ page }) => {
  const sink = attachSink(page)
  const pageviews = (path: string) =>
    sink.umami.filter((b) => b.type === 'event' && !b.name && new URL(b.url ?? "", "http://x").pathname === path).length

  await page.goto('/welcome')
  await sink.waitForEvent('landing_view')
  await page.getByTestId('landing-cta-primary').click()
  await page.waitForURL('**/login**')
  await sink.waitForEvent('screen_view')
  await sink.settle()
  // Give a stacked wrapper's delayed send (the tracker defers page views by 300ms) time to show up.
  await page.waitForTimeout(1_500)

  expect(pageviews('/welcome'), 'page views for /welcome').toBe(1)
  expect(pageviews('/login'), 'page views for /login').toBe(1)
  expect(await page.locator('script[data-umami-installed]').count(), 'tracker tags').toBe(1)
})
