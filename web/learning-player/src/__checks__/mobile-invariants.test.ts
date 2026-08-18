import { describe, expect, it } from 'vitest'
import apiSrc from '../services/api.ts?raw'
import playerStoreSrc from '../stores/player.ts?raw'
import playerViewSrc from '../views/PlayerView.vue?raw'
import highlightsViewSrc from '../views/HighlightsView.vue?raw'
import mainSrc from '../main.ts?raw'
import authStoreSrc from '../stores/auth.ts?raw'
import nativeSrc from '../services/native.ts?raw'
import tierSrc from '../services/tier.ts?raw'
import indexHtml from '../../index.html?raw'
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
    expect(highlightsViewSrc, 'HighlightsView must branch on isNative() for export').toMatch(
      /isNative\(\)/,
    )
    expect(highlightsViewSrc).toMatch(/saveAndShareText/)
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
