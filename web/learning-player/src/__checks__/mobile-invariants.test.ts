import { describe, expect, it } from 'vitest'
import apiSrc from '../services/api.ts?raw'
import playerStoreSrc from '../stores/player.ts?raw'
import playerViewSrc from '../views/PlayerView.vue?raw'
import highlightsViewSrc from '../views/HighlightsView.vue?raw'
import exportViewerSrc from '../components/ExportViewer.vue?raw'
import mainSrc from '../main.ts?raw'
import analyticsSrc from '../services/analytics.ts?raw'
import authStoreSrc from '../stores/auth.ts?raw'
import nativeSrc from '../services/native.ts?raw'
import tierSrc from '../services/tier.ts?raw'
import indexHtml from '../../index.html?raw'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
// Read from disk: a `?raw` import of a .css file comes back EMPTY under vitest, which does not
// process CSS — a check against it would match nothing and pass on a deleted rule.
const styleSrc = readFileSync(resolve(__dirname, '..', 'style.css'), 'utf8')
import capacitorConfigSrc from '../../capacitor.config.ts?raw'
import iosInfoPlist from '../../ios/App/App/Info.plist?raw'
import iosAppDelegate from '../../ios/App/App/AppDelegate.swift?raw'
import iosAuthSession from '../../ios/App/App/AuthSession.swift?raw'
import iosMainVC from '../../ios/App/App/MainViewController.swift?raw'
import androidManifest from '../../android/app/src/main/AndroidManifest.xml?raw'

/**
 * Guardrail (#1311) — codified mobile-readiness invariants that must not regress. These are static
 * source checks that run in the normal unit suite (so CI gates them), guarding the Capacitor-shell
 * prerequisites established in the mobile epic (#1298). If one fails, a change re-broke a mobile
 * invariant — fix the change, don't weaken the check.
 */

// All .vue components, loaded as raw source.
const components = import.meta.glob('../**/*.vue', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

describe('mobile invariants (guardrail #1311)', () => {
  it('API base resolves from VITE_API_BASE_URL (reaches the API in the capacitor:// shell)', () => {
    // A bare origin-relative base fails in the WebView (origin = capacitor://localhost).
    expect(apiSrc, 'services/api.ts must resolve the base from VITE_API_BASE_URL').toMatch(
      /VITE_API_BASE_URL/,
    )
  })

  it('index.html sets viewport-fit=cover (enables env(safe-area-inset-*))', () => {
    expect(indexHtml).toMatch(/viewport-fit=cover/)
  })

  it('the iOS webview does NOT also inset for the safe area (#2004 item 1)', () => {
    // `viewport-fit=cover` + `env(safe-area-inset-*)` is the app's inset mechanism, asserted by the
    // test above. The native setting makes WKWebView apply the SAME inset again, so it is paid
    // twice.
    //
    // Where, exactly, was measured on an iPhone 17 Pro simulator — same build, only this value
    // changed, screenshots diffed row by row: the header and the entire page body came out
    // BYTE-IDENTICAL, and only the bottom nav region moved. So the double payment is at the
    // BOTTOM: the nav is lifted off the screen edge and a dead band is left beneath it that the
    // app never paints. An earlier version of this comment claimed the doubling was at the top,
    // above the brand bar; that was an untested inference and the measurement falsifies it.
    //
    // The two settings are a pair: whoever changes one has to answer for the other, so they are
    // guarded together rather than left to be rediscovered on a device.
    expect(capacitorConfigSrc).toMatch(/contentInset:\s*'never'/)
    expect(
      capacitorConfigSrc.replace(/\/\*[\s\S]*?\*\//g, ''),
      "contentInset: 'always' double-pays the inset the CSS already applies",
    ).not.toMatch(/contentInset:\s*'always'/)
  })

  it('no raw vh units in components — use dvh (iOS 100vh over-report clips sheets/footers)', () => {
    const offenders: string[] = []
    for (const [path, src] of Object.entries(components)) {
      const matches = src.match(/\[[^\]]*vh\]/g) ?? [] // Tailwind arbitrary values e.g. max-h-[85vh]
      for (const m of matches) {
        if (!/[dsl]vh\]/.test(m)) offenders.push(`${path}: ${m}`) // allow dvh / svh / lvh
      }
      if (/\bmin-h-screen\b/.test(src)) offenders.push(`${path}: min-h-screen (use min-h-dvh)`)
    }
    expect(offenders, `raw vh usage found:\n${offenders.join('\n')}`).toEqual([])
  })

  it('playback state lives in the player store, not PlayerView local refs', () => {
    // Re-adding local playback refs would break MediaSession / native controls (one source of truth).
    for (const ref of ['playing', 'currentTime', 'duration', 'rate']) {
      expect(
        playerViewSrc,
        `PlayerView must not declare a local '${ref}' ref — it belongs to stores/player.ts`,
      ).not.toMatch(new RegExp(`const\\s+${ref}\\s*=\\s*ref\\(`))
    }
    expect(playerViewSrc).toMatch(/usePlayerStore\(\)/)
  })

  it('MediaSession is wired in the player store (lock-screen / headphone controls)', () => {
    expect(playerStoreSrc).toMatch(/navigator\.mediaSession/)
    expect(playerStoreSrc).toMatch(/setActionHandler/)
  })
})

describe('native-shell invariants (guardrail #1310)', () => {
  it('highlights export has a native (write+share) path — <a download> cannot save in WKWebView', () => {
    // The export's formats live in the shared ExportViewer now (operator 2026-10-05), so the
    // native branch is asserted THERE, and HighlightsView must actually route through it.
    expect(highlightsViewSrc, 'HighlightsView must export through ExportViewer').toMatch(/<ExportViewer/)
    expect(exportViewerSrc, 'ExportViewer must branch on isNative() for export').toMatch(
      /isNative\(\)/,
    )
    expect(exportViewerSrc).toMatch(/saveAndShareText/)
  })

  it('telemetry tags the platform (web|ios|android) so native builds stay separable', () => {
    expect(mainSrc).toMatch(/platform:\s*platform\(\)/)
  })

  it('iOS background audio is configured (UIBackgroundModes audio + AVAudioSession playback)', () => {
    // Both halves are required — the plist mode alone does nothing without an active playback session.
    expect(iosInfoPlist, 'Info.plist must declare the audio background mode').toMatch(
      /<key>UIBackgroundModes<\/key>[\s\S]*?<string>audio<\/string>/,
    )
    expect(iosAppDelegate, 'AppDelegate must set an AVAudioSession playback category').toMatch(
      /AVAudioSession[\s\S]*?setCategory\(\.playback/,
    )
  })

  it('native OAuth: bearer plumbing + browser login + deep-link scheme are all wired', () => {
    // API client carries the bearer token (external OAuth browser can't hand the cookie to the WebView).
    expect(apiSrc, 'api.ts must set an Authorization: Bearer header from setAuthToken').toMatch(
      /Authorization/,
    )
    expect(apiSrc).toMatch(/setAuthToken/)
    // Auth store starts native login instead of a dead-end WebView redirect.
    expect(authStoreSrc).toMatch(/isNative\(\)/)
    expect(authStoreSrc).toMatch(/startNativeLogin/)
    // iOS returns prompt-free via ASWebAuthenticationSession (no "Open in app?" dialog); Android via
    // the intent-filter callback. native.ts must branch iOS → AuthSession.
    expect(iosAuthSession, 'AuthSession.swift must use ASWebAuthenticationSession').toMatch(
      /ASWebAuthenticationSession/,
    )
    // App-embedded plugins aren't in capacitor.config.json's packageClassList, so they must be
    // registered explicitly in the bridge VC — without this the plugin is "not implemented on ios".
    expect(iosMainVC, 'MainViewController must register the AuthSession plugin instance').toMatch(
      /registerPluginInstance\(AuthSession\(\)\)/,
    )
    expect(iosAuthSession, 'AuthSession must be a CAPBridgedPlugin (registerPluginInstance requires it)').toMatch(
      /CAPBridgedPlugin/,
    )
    expect(nativeSrc, 'native.ts must route iOS login through the AuthSession plugin').toMatch(
      /AuthSession/,
    )
    // The closelistening:// scheme must be registered on BOTH platforms or the callback can't return.
    expect(iosInfoPlist, 'iOS Info.plist must register the closelistening URL scheme').toMatch(
      /<key>CFBundleURLSchemes<\/key>[\s\S]*?<string>closelistening<\/string>/,
    )
    expect(androidManifest, 'AndroidManifest must register the closelistening deep-link scheme').toMatch(
      /android:scheme="closelistening"/,
    )
  })

  it('Android background audio: foreground media service + permission declared, wired to play/pause', () => {
    expect(androidManifest, 'manifest must declare the PlaybackService as mediaPlayback FGS').toMatch(
      /android:name="\.PlaybackService"[\s\S]*?android:foregroundServiceType="mediaPlayback"/,
    )
    expect(androidManifest).toMatch(/FOREGROUND_SERVICE_MEDIA_PLAYBACK/)
    // The store must start/stop the keep-alive on play/pause or backgrounded audio dies.
    expect(playerStoreSrc).toMatch(/startBackgroundAudio\(\)/)
    expect(playerStoreSrc).toMatch(/stopBackgroundAudio\(\)/)
  })

  it('dev/prod tier switch: api base resolves per-tier, no staging, prod-locked in release', () => {
    // The API base flows through the tier resolver, not a bare env const.
    expect(apiSrc).toMatch(/resolveApiBase\(\)/)
    // Podcast has no staging (ADR-126): the Tier type is only dev + prod.
    expect(tierSrc).toMatch(/type Tier = 'dev' \| 'prod'/)
    // Release is prod-locked via the build flag; the switch is native + internal only.
    expect(tierSrc).toMatch(/__MOBILE_INTERNAL__/)
    expect(tierSrc).toMatch(/isNativePlatform\(\)/)
    // Telemetry follows the tier (Sentry), Umami stays prod (unified) — main switches Sentry by tier.
    expect(mainSrc).toMatch(/nativeDevTier/)
  })

  it('the Umami tag excludes the query string, so search terms never reach analytics (#2264)', () => {
    // Umami auto-tracks the FULL URL, and the search term travels in the query string. Five call
    // sites put it there — SearchView.vue (x3), BrowseView.vue, HomeView.vue, and LibraryView.vue,
    // where a saved-search link replays a stored term back into the URL. So for a while every
    // search a user ran was recorded, verbatim, as part of a page-view URL.
    //
    // Asserted on the injection site rather than on a rendered DOM because this is the only place
    // the attribute can come from, and a static check cannot be made to pass by a mock. The site is
    // `services/analytics.ts` (`installUmami`) — it was an inline block in main.ts until sign-out
    // needed to replace the tracker, and main.ts now only calls it.
    expect(analyticsSrc).toMatch(/setAttribute\(\s*['"]data-exclude-search['"]/)
    expect(mainSrc).toMatch(/installUmami\(\)/)

    // The attribute is only meaningful on the tag that actually gets injected, so prove it sits
    // between the website id and the append rather than somewhere unreachable.
    const injection = analyticsSrc.slice(analyticsSrc.indexOf('data-website-id'))
    expect(
      injection.slice(0, injection.indexOf('appendChild')),
      'data-exclude-search must be set on the injected Umami script, before it is appended',
    ).toContain('data-exclude-search')

    // ONE injection path. Two tags double-count every page view, and the reason the logic moved out
    // of main.ts was precisely that a second copy would drift.
    expect(
      mainSrc.includes("createElement('script')") && mainSrc.includes('data-website-id'),
      'main.ts must not inject its own Umami tag — call installUmami()',
    ).toBe(false)
  })

  it('Sentry scrubs the query string, so search terms never reach GlitchTip either (#2264)', () => {
    // GlitchTip is a SECOND sink for the same text, and Umami's data-exclude-search does nothing
    // for it. `sendDefaultPii: false` does not cover it: that governs IP/cookies/user data, not
    // query strings. The leak path, traced through the installed SDK (@sentry/vue 10.60.0), is
    // navigation breadcrumbs — `breadcrumbs.js` sets from/to to `parseUrl(...).relative`, which
    // `@sentry/core/src/utils/url.ts` defines as `path + query + fragment`.
    //
    // This only checks the WIRING. The scrubbing behaviour is tested directly in
    // `services/telemetryScrub.test.ts`, which is the half that can actually be wrong — a hook
    // that exists and strips nothing would satisfy any source-text check.
    expect(mainSrc).toMatch(/beforeBreadcrumb:\s*scrubNavigationBreadcrumb/)
    expect(mainSrc).toMatch(/beforeSend:\s*scrubEventRequestUrl/)
    expect(mainSrc).toMatch(/from '\.\/services\/telemetryScrub'/)
  })

  it('every surface pinned to the bottom edge reserves the bottom inset (operator 2026-10-05)', () => {
    /*
     * Android 15+ draws the app edge-to-edge, UNDER its navigation bar — three buttons on a
     * transparent strip. A beta tester scrolled an entity card to its end and could see the note's
     * Save button through the bar but not press it: the shared sheet reserved nothing at the bottom.
     * The iPhone home indicator covers the same strip.
     *
     * The rule: anything that reaches the bottom of the screen reserves `safe-area-inset-bottom`
     * (the device's own height for that bar), or is named here with the reason it need not.
     */
    const EXEMPT: Record<string, string> = {
      '../components/AppSplash.vue': 'a full-screen image with no controls; it reserves its own text line',
      '../components/AvatarCropModal.vue': 'a CENTRED dialog with padding — it never touches the edge',
    }
    const pinned = Object.entries(components).filter(([, src]) =>
      /class="[^"]*\bfixed\b[^"]*\b(inset-0|bottom-0)\b/.test(src),
    )
    expect(pinned.length, 'the scan found nothing — the pattern is broken').toBeGreaterThan(3)
    const missing = pinned
      .filter(([path, src]) => !(path in EXEMPT) && !src.includes('safe-area-inset-bottom'))
      .map(([path]) => path)
    expect(missing, 'pinned to the bottom edge with no bottom inset').toEqual([])

    // The shared bottom SHEET (topic / person / storyline / theme cards) is styled in CSS, not in a
    // class list, so it is checked where it is defined: the phone rule pads, the centred one does not.
    const sheet = styleSrc.match(/\.lp-sheet \{[^}]*\}/)?.[0] ?? ''
    expect(sheet, 'the phone .lp-sheet rule must reserve the bottom inset').toMatch(
      /padding-bottom:\s*env\(safe-area-inset-bottom\)/,
    )
  })

  it('the bottom nav clears the home indicator and does not trap page content (#1594)', () => {
    const nav = components['../components/BottomNav.vue'] ?? ''
    expect(nav, 'BottomNav.vue must exist').not.toBe('')

    // Fixed to the bottom, so it MUST respect the iOS home indicator or the last tab sits under it.
    expect(nav).toContain('safe-area-inset-bottom')
    // Mobile only — the desktop header nav already covers those widths.
    expect(nav).toContain('sm:hidden')

    // A fixed bar covers page content unless the scroll container reserves room for it. Without
    // this, the last item of every list is unreachable on a phone — the classic bottom-nav bug.
    //
    // This used to assert the literal strings `pb-24` and `sm:pb-6`, which is precisely why the
    // geometry could be wrong while the check stayed green: 96px of mobile padding against a ~52px
    // tab bar PLUS a ~62px mini-player, and 24px on desktop against that same mini-player. The
    // classes were present and the content was still covered. Assert the two properties that
    // actually matter instead — the reservation accounts for the safe-area inset, and it RESPONDS
    // to whether the mini-player is on screen rather than being a constant.
    const app = components['../App.vue'] ?? ''
    expect(app, 'App.vue must exist').not.toBe('')
    expect(app, 'main must reserve space for the fixed bars').toMatch(
      /:class="mainBottomPadding"/,
    )
    expect(app, 'the reservation must clear the home indicator').toMatch(
      /mainBottomPadding[\s\S]{0,400}safe-area-inset-bottom/,
    )
    expect(app, 'the reservation must depend on whether the mini-player is showing').toMatch(
      /mainBottomPadding[\s\S]{0,200}player\.currentSlug/,
    )
  })

  it('a phone shows exactly ONE navigation system (#1594 follow-up)', () => {
    // The bottom tab bar shipped without hiding the header icon links, so mobile carried both at
    // once: Search twice, Library and Profile at the top AND bottom of the same screen. Two navs
    // read as two designs stacked, and they spend the scarcest space on a phone twice over.
    const app = components['../App.vue'] ?? ''
    expect(app, 'App.vue must exist').not.toBe('')

    // The icon-link group is desktop-only...
    expect(app, 'the header icon links must be hidden below sm').toMatch(
      /class="hidden items-center gap-1\.5 sm:flex"/,
    )
    // ...and the tab bar is mobile-only, so the two never coexist.
    expect(components['../components/BottomNav.vue'] ?? '').toContain('sm:hidden')
  })
})
