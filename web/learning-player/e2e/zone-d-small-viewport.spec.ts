import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Zone D must survive its worst real case on the smallest phone we support (#1978 follow-up).
 *
 * ## Why 375 x 667 and why this episode
 *
 * The design harness runs at Pixel 7 (412 CSS px). 375pt is the iPhone SE / mini width and the
 * narrowest target — 9% less room, which is exactly where a panel sized by eye at 412 starts
 * clipping. Nothing tested it.
 *
 * The insight is not invented. Measured across the 36-episode fixture corpus: 124 Insight nodes,
 * of which 108 are renderable in Zone D (the other 16 have only untimed or degenerate quote
 * windows and can never surface). The longest RENDERABLE one is 200 characters, in p06_e02 "More
 * Drift, Less Signal" at 12s. That is the worst case the panel can actually be asked to draw, so
 * it is the one asserted here rather than a synthetic string that proves nothing about the corpus.
 *
 * Note the overall maximum (200) and the renderable maximum (200) coincide, but only by luck: the
 * longest insight overall has no timed quote at all, and a different corpus could easily put its
 * longest text somewhere Zone D never reaches.
 */
test.use({ viewport: { width: 375, height: 667 } })

test('Zone D renders the corpus-longest insight at 375pt without clipping', async ({ page }, testInfo) => {
  await signInIsolated(page, 'zone-d-375', testInfo)

  await page.goto('/podcast/p06')
  await page.waitForLoadState('networkidle')
  await page.getByRole('link', { name: /More Drift, Less Signal/i }).first().click()
  await page.waitForURL(/\/episode\//)
  await page.waitForLoadState('networkidle')

  /**
   * Sweep the whole episode rather than pinning one timestamp.
   *
   * The first version seeked to 13s, computed offline from the graph as the moment the 200-char
   * insight is live. A different insight was showing — the timeline the app derives is not the one
   * reconstructed from `.gi.json` by hand, and a test that depends on my arithmetic about the
   * fixture tests my arithmetic. Stepping through and asserting the invariant for EVERY insight the
   * panel actually renders is both stronger and free of that assumption.
   *
   * The playhead is driven directly: real audio decode is not needed for a layout assertion, and
   * the fixture MP3 does not reliably decode headlessly (see full-listen.spec.ts).
   */
  /**
   * Wait until the <audio> element ACCEPTS a seek before sweeping.
   *
   * Until the element has loaded enough to be seekable, `currentTime = t` is silently dropped — the
   * write does not throw and does not stick, so the sweep samples an unmoved playhead and sees
   * whichever insight is live near zero. That made this test flaky rather than wrong: one run
   * observed a 200-char insight and passed, the next observed only 147 and failed its own vacuity
   * guard. Reading the value back is the only way to know the seek landed.
   */
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const audio = document.querySelector('audio')
          if (!audio) return -1
          audio.currentTime = 30
          return audio.currentTime
        }),
      { timeout: 20_000, message: 'the <audio> element never became seekable' },
    )
    .toBeGreaterThan(1)

  const observed: Array<{ t: number; chars: number; clipped: boolean; right: boolean; bottom: boolean; left: number }> = []
  for (let t = 0; t <= 249; t += 4) {
    const landed = await page.evaluate((time) => {
      const audio = document.querySelector('audio')
      if (!audio) return -1
      audio.currentTime = time
      audio.dispatchEvent(new Event('timeupdate'))
      return audio.currentTime
    }, t)
    // A step whose seek did not take tells us nothing about that moment; skip rather than record a
    // sample labelled with a time the player was never at.
    if (Math.abs(landed - t) > 1) continue
    await page.waitForTimeout(20) // let Vue apply the new active insight before measuring
    const sample = await page.evaluate(() => {
      const el = document.querySelector<HTMLElement>('[data-testid="player-zone-d-live"]')
      if (!el) return null
      const txt = el.querySelector<HTMLElement>('p[class*="line-clamp"]')
      if (!txt) return null
      const r = el.getBoundingClientRect()
      if (r.width === 0) return null
      return {
        chars: (txt.textContent || '').trim().length,
        clipped: txt.scrollHeight > txt.clientHeight + 1,
        right: r.right > window.innerWidth + 1,
        bottom: r.bottom > window.innerHeight + 1,
        left: r.left,
      }
    })
    if (sample) observed.push({ t, ...sample })
  }

  // Guards the test itself. "Nothing was clipped" is trivially true if the panel never appeared, or
  // if every insight it showed was short — either way this file would stop testing its own name.
  expect(observed.length, 'Zone D never rendered at any point in the episode').toBeGreaterThan(0)
  const longest = Math.max(...observed.map((o) => o.chars))
  expect(
    longest,
    `the longest insight Zone D showed was ${longest} chars; the corpus contains a 200-char one, ` +
      'so this sweep is not exercising the worst case it claims to',
  ).toBeGreaterThan(150)

  const clipped = observed.filter((o) => o.clipped)
  expect(
    clipped.map((o) => `t=${o.t}s (${o.chars} chars)`),
    'these insights are truncated by line-clamp at 375pt',
  ).toEqual([])

  const overflow = observed.filter((o) => o.right || o.bottom || o.left < 0)
  expect(
    overflow.map((o) => `t=${o.t}s right=${o.right} bottom=${o.bottom} left=${o.left}`),
    'the panel leaves the 375pt viewport at these moments',
  ).toEqual([])
})
