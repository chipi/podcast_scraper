import { expect, test } from '@playwright/test'
import { attachSink, DEV_GLITCHTIP_PROJECT_ID } from './sink'
import './settle'

/**
 * The SECOND telemetry sink, and the one the privacy fix was nearly incomplete for (#2264).
 *
 * Umami's `data-exclude-search` keeps the search term out of Umami. It does nothing for GlitchTip,
 * and `sendDefaultPii: false` does not cover it either — that option governs IP address, cookies and
 * user data, not query strings. The search term reaches GlitchTip through NAVIGATION BREADCRUMBS,
 * which record `from`/`to` as path + query + fragment, and through the error's own `request.url`.
 *
 * So an error thrown anywhere after a search would have carried the listener's search text into the
 * error tracker, attached to a stack trace, retained on whatever schedule the error tracker keeps.
 * That is a different and worse exposure than a page view.
 *
 * The hooks are `beforeBreadcrumb: scrubNavigationBreadcrumb` and `beforeSend: scrubEventRequestUrl`,
 * unit-tested in `services/telemetryScrub.ts`. This proves they hold on the real wire, against the
 * real GlitchTip, including the fields nobody thought to look at — the leak was found in
 * `breadcrumbs[].data.to` precisely because that field was not on anyone's list.
 */

/** A string nothing else in the corpus can produce, so a hit is unambiguous. */
const PROBE = 'leak_probe_gt_7f3a1c9e'

test.describe('dev routing', () => {
  test('errors go to the DEV GlitchTip project, tagged as the player, in environment=dev', async ({
    page,
  }) => {
    const sink = attachSink(page)
    await page.goto('/welcome')
    await page.waitForLoadState('networkidle')

    // A genuine uncaught error: thrown from a timer so it reaches `window.onerror` the way a real
    // bug would, rather than being handed to the SDK by the test.
    await page.evaluate(() => {
      setTimeout(() => {
        throw new Error('telemetry tier: deliberate uncaught error for GlitchTip routing proof')
      }, 0)
    })

    // The ERROR envelope specifically. The SDK also sends a session envelope, which normally arrives
    // first and contains neither tags nor `environment` — asserting on `sentry[0]` inspected that one.
    const env = await sink.waitForError(1, 30_000)

    // WHICH project, read off the ingest URL. Asserted on the URL rather than `header.dsn`, because
    // modern SDKs put the DSN in the request URL and omit the header field entirely — so a
    // `header.dsn` assertion compares against an empty string and passes for the wrong reason.
    expect(env.url, 'the envelope must be POSTed to the dev project').toContain(
      `/api/${DEV_GLITCHTIP_PROJECT_ID}/envelope`,
    )
    expect(env.url, 'must NOT be prod project 5, which the deployed player reports into').not.toContain(
      '/api/5/envelope',
    )

    const body = env.rawText
    expect(body).toContain('"environment":"dev"')
    // `component` keeps the player's stream separable from api / pipeline / viewer, and `platform`
    // separates the native shells from the web player in the same stream.
    expect(body).toContain('"component":"player"')
  })
})

test.describe('the ?q= scrub', () => {
  test('a search term never reaches GlitchTip, in ANY field of the envelope', async ({ page }) => {
    const sink = attachSink(page)

    // Walk through a search URL so a navigation breadcrumb is recorded carrying the query, then
    // navigate again so the breadcrumb is a `from`→`to` pair, then throw. This is the exact shape that
    // leaked: the error itself happens somewhere innocent, and the search term rides along in history.
    await page.goto(`/search?q=${PROBE}&scope=corpus`)
    await page.waitForLoadState('networkidle')
    await page.goto('/welcome')
    await page.waitForLoadState('networkidle')

    await page.evaluate(() => {
      setTimeout(() => {
        throw new Error('telemetry tier: deliberate uncaught error for the scrub proof')
      }, 0)
    })

    await sink.waitForError(1, 30_000)

    // 1. The blunt check: the probe appears NOWHERE in any captured envelope.
    for (const env of sink.sentry) {
      expect(
        env.rawText,
        'the search term must not appear anywhere in a GlitchTip envelope',
      ).not.toContain(PROBE)
    }

    // 2. The structural check, over every URL-ish string anywhere in the payload. This flattens the
    //    whole object rather than inspecting the two fields the fix touches, because the leak was
    //    found in a field nobody was inspecting. A test that only looks where the fix looked cannot
    //    catch the next one.
    const urls = sink.sentryUrlStrings()
    expect(urls.length, 'the envelope should carry some URLs, or this proves nothing').toBeGreaterThan(0)
    for (const u of urls) {
      expect(u, `a URL string still carried the probe: ${u}`).not.toContain(PROBE)
    }

    // 3. The path itself must SURVIVE. A scrub that deleted the URL would also pass checks 1 and 2
    //    while destroying the only thing that makes an error diagnosable.
    expect(
      urls.some((u) => u.includes('/welcome') || u.includes('/search')),
      'the PATH must still be reported — scrubbing the query must not scrub the route',
    ).toBe(true)
  })

  test('Umami and GlitchTip are scrubbed by DIFFERENT mechanisms, and both hold', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto(`/search?q=${PROBE}&scope=corpus`)
    await page.waitForLoadState('networkidle')
    await page.evaluate(() => {
      setTimeout(() => {
        throw new Error('telemetry tier: deliberate uncaught error, both-sinks proof')
      }, 0)
    })
    await sink.waitForError(1, 30_000)

    // Umami's side is the `data-exclude-search` attribute on the script tag.
    for (const u of sink.reportedUrls()) {
      expect(u, `a Umami beacon carried the query: ${u}`).not.toContain(PROBE)
    }
    // GlitchTip's side is the two SDK hooks. Two sinks, two mechanisms, one requirement — which is
    // exactly why fixing only the Umami half looked complete and was not.
    for (const env of sink.sentry) {
      expect(env.rawText).not.toContain(PROBE)
    }
  })
})
