import { expect, test } from '@playwright/test'
import { attachSink, DEV_UMAMI_WEBSITE_ID } from './sink'

/**
 * The harness check. If this fails, every richer assertion below it is measuring nothing.
 *
 * It exists because the failure mode this whole arc is about is a telemetry path that LOOKS wired
 * and silently goes nowhere. A suite that asserts "the app called track()" would have passed every
 * day that the hardcoded website id pointed at a website which did not exist. So the first thing
 * proven here is not a property of an event — it is that the pipe is real and points at the DEV
 * target.
 */
test('the beacon is installed, reaches the DEV Umami site, and is accepted', async ({ page }) => {
  const sink = attachSink(page)
  await page.goto('/welcome')

  const view = await sink.waitForEvent('landing_view')

  // 1. It went to the dev site, not the prod one. Prod `cd384a3e-…` carries 562 real events; a test
  //    run that wrote there would corrupt the numbers the beta is judged on.
  expect(view.website, 'events must report into the DEV website id').toBe(DEV_UMAMI_WEBSITE_ID)

  // 2. The script tag is actually present and carries the search-excluding attribute.
  const tag = page.locator('script[data-website-id]')
  await expect(tag).toHaveCount(1)
  expect(await tag.getAttribute('data-website-id')).toBe(DEV_UMAMI_WEBSITE_ID)
  expect(
    await tag.getAttribute('data-exclude-search'),
    'data-exclude-search is what keeps the search term out of tracked URLs',
  ).not.toBeNull()

  // 3. Umami ACCEPTED it. A 200 is not enough: the collector answers 200 with {"beep":"boop"} when
  //    it drops a request as a bot, and 200 with {"error":{"message":"Website not found."}} for an
  //    unknown website id. Both are how dev analytics managed to look healthy while storing nothing.
  const resp = await page.request.post('http://127.0.0.1:3001/api/send', {
    headers: { 'Content-Type': 'application/json' },
    data: {
      type: 'event',
      payload: {
        website: DEV_UMAMI_WEBSITE_ID,
        hostname: '127.0.0.1',
        url: '/__harness__',
        name: 'harness_probe',
        language: 'en-US',
        screen: '1280x720',
      },
    },
  })
  expect(resp.status()).toBe(200)
  const text = await resp.text()
  expect(text, 'bot-drop: the collector stored nothing').not.toContain('beep')
  expect(text, 'the dev website id must exist in the instance').not.toContain('Website not found')
})
