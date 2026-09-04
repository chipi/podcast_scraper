import { expect, test } from '@playwright/test'
import { openTranscript } from './helpers'

/**
 * Guardrail (#1312) — runtime mobile invariants a real browser can prove that the static source
 * checks (src/__checks__/mobile-invariants.test.ts, #1311) cannot: the sticky controls actually
 * stay pinned while the transcript scrolls under them, MediaSession is wired into the REAL view
 * lifecycle (metadata set on load, playbackState tracks play/pause via navigator.mediaSession), and
 * the canvas paints dark (no white flash). These gate the Capacitor-shell prerequisites (#1298)
 * against regression under CI (make test-app-e2e). If one fails, a change re-broke a mobile
 * invariant — fix the change, don't weaken the check.
 *
 * The fixture audio is real and decodable (#1618) — the corpus points at the mock podcast
 * host, so the transport panel renders with no interception anywhere in this suite.
 */

// Newest fixture episode — same one transcript.spec / capture.spec / full-listen use.
const EPISODE_TITLE = 'Index Investing Without the Myths'

test.describe('mobile invariants (guardrail #1312)', () => {
  test('sticky controls stay pinned while the transcript scrolls under them', async ({
    page,
  }, testInfo) => {
    // Mobile-only: on desktop the controls are static in the left column (the sticky wrapper is
    // inert), so "stays pinned on scroll" is not a desktop behaviour.
    test.skip(testInfo.project.name !== 'mobile-chrome', 'sticky controls are a mobile behaviour')

    await page.goto('/podcast/p05')
    await page.getByText(EPISODE_TITLE).first().click()
    await page.getByRole('heading', { name: new RegExp(EPISODE_TITLE) }).waitFor()
    await openTranscript(page) // reveal the transcript so there's content to scroll under the controls

    const sticky = page.getByTestId('player-controls-sticky')
    await expect(sticky).toBeInViewport()

    // Scroll the page down past the fold; the transcript flows under the controls.
    await page.evaluate(() => window.scrollTo(0, 600))
    await expect(async () => {
      expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(200)
    }).toPass()

    // The controls must remain in the viewport (sticky top-0), not scroll off with the page.
    await expect(sticky).toBeInViewport()
    // And within the controls, Play/Pause stays reachable.
    await expect(
      page
        .getByRole('button', { name: 'Play', exact: true })
        .or(page.getByRole('button', { name: 'Pause', exact: true })),
    ).toBeInViewport()
  })

  test('MediaSession metadata is set on load and playbackState tracks play/pause', async ({
    page,
  }) => {
    await page.goto('/podcast/p05')
    await page.getByText(EPISODE_TITLE).first().click()
    await page.getByRole('heading', { name: new RegExp(EPISODE_TITLE) }).waitFor()

    // The view calls player.setMetadata() on load → the real navigator.mediaSession carries the
    // episode's lock-screen metadata (jsdom unit tests can only stub this; a real browser proves it).
    await expect(async () => {
      const title = await page.evaluate(() => navigator.mediaSession?.metadata?.title ?? null)
      expect(title).toBe(EPISODE_TITLE)
    }).toPass({ timeout: 15_000 })

    // Asserted, not guarded. The "routed WAV" the old comment referred to is gone — since #1618
    // this tier serves the REAL fixture mp3 from the mock host, and the sticky-controls test in
    // this same file already asserts Play/Pause renders in this very project. So a missing Play
    // button here is a regression, not an engine limitation. The old `if (!visible) skip` made
    // the playbackState assertions below unreachable without anything ever going red.
    const play = page.getByRole('button', { name: 'Play', exact: true })
    await expect(play).toBeVisible()

    await play.click()
    // onPlay (bound to the <audio> @play event) sets mediaSession.playbackState.
    await expect(async () => {
      const state = await page.evaluate(() => navigator.mediaSession?.playbackState ?? 'none')
      expect(state).toBe('playing')
    }).toPass({ timeout: 5_000 })

    await page.getByRole('button', { name: 'Pause', exact: true }).click()
    await expect(async () => {
      const state = await page.evaluate(() => navigator.mediaSession?.playbackState ?? 'none')
      expect(state).toBe('paused')
    }).toPass({ timeout: 5_000 })
  })

  test('canvas paints dark — no white flash', async ({ page }) => {
    await page.goto('/podcast/p05')
    // The app is dark-first (UXS): the document background must not be white, or a native shell
    // shows a white flash on launch / between views. The dark canvas is painted on <body>
    // (style.css → background: var(--lp-canvas)); that's the surface behind every view.
    const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor)
    // Parse rgb() → reject near-white (all channels high).
    const m = bg.match(/(\d+),\s*(\d+),\s*(\d+)/)
    expect(m, `expected a resolved rgb background, got "${bg}"`).not.toBeNull()
    const [r, g, b] = m!.slice(1).map(Number)
    const isNearWhite = r > 230 && g > 230 && b > 230
    expect(isNearWhite, `page background ${bg} is near-white (white-flash risk)`).toBe(false)
  })
})
