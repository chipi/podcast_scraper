import { createApp } from 'vue'
import { createPinia } from 'pinia'
import * as Sentry from '@sentry/capacitor'
import * as SentryVue from '@sentry/vue'
import './style.css'
import App from './App.vue'
import { router } from './router'
import { i18n } from './i18n'
import { applyTheme } from './theme/theme'
import { applyDirection, resolveDirection } from './theme/direction'
import { initGateCookie, platform, rehydrateNativeToken } from './services/native'
import { getTier, tierSwitchEnabled } from './services/tier'
import { setOnUnauthorized } from './services/api'
import { installUmami } from './services/analytics'
import { installLastPlace, recordPlace, type SavedEpisode } from './services/lastPlace'
import { initLifecycle } from './services/lifecycle'
import { scrubEventRequestUrl, scrubNavigationBreadcrumb } from './services/telemetryScrub'
import { localSourceFor } from './services/downloads'
import { useAuthStore } from './stores/auth'
import { useDownloadsStore } from './stores/downloads'
import { usePlayerStore } from './stores/player'

applyTheme('dark')

// Expose build identity for update-path debugging. When a user reports
// 'the PWA isn't updating', the running client's sha + time can be read
// from window.__buildInfo (DevTools console or a support form) to
// distinguish 'stuck client' from 'cache never invalidated'. See
// vite.config.ts `define:` block for the injection.
window.__buildInfo = { sha: __BUILD_SHA__, time: __BUILD_TIME__ }

console.info(`[app] Learning Player build=${__BUILD_SHA__} time=${__BUILD_TIME__}`)

// Visual-direction switch (#1949). Logic and rationale live in `theme/direction.ts`, where it is
// reachable from a test; this is only the wiring to the real URL, storage and document.
applyDirection(
  document.documentElement,
  resolveDirection(window.location.search, sessionStorage),
)

const app = createApp(App)

// Sentry/GlitchTip init for the consumer player — mirrors the viewer
// (web/gi-kg-viewer/src/main.ts). Gated on ``VITE_SENTRY_DSN_PLAYER`` so the
// default (no DSN) stays a true no-op for dev / CI / any build without the
// build-arg. The DSN reaches Vite at build time (baked into the bundle); the
// docker build passes it via ``--build-arg VITE_SENTRY_DSN_PLAYER=...``. Points
// at the self-hosted GlitchTip through the public ingest edge — the player is a
// browser client, so it can't reach the tailnet-only backend directly.
// Unlike the viewer, the player nginx does no runtime ``sub_filter`` env
// injection, so ``environment`` comes from the build mode and ``release`` from
// the existing ``__BUILD_SHA__`` define.
// Dev rung of the env ladder (dev → prod; the player has no staging). In `vite
// dev` (desktop, http origin) errors go to the dedicated `player-dev` GlitchTip
// project via the Tailscale host `homelab` — NO fixed IP, tailnet-only; a
// stranger who runs the repo reports nothing (the transport silently fails). The
// key is a public browser id (ships in the bundle) — safe to commit.
// `VITE_ANALYTICS_OFF=1` disables the default.
//
// On a NATIVE device the WebView origin is https, so the plain-http default is
// blocked (iOS ATS + Android mixed-content). For dev-tier o11y on-device, expose
// GlitchTip over https on the tailnet (`tailscale serve`, see the capacitor build
// runbook) and inject that https DSN via VITE_SENTRY_DSN_PLAYER_DEV in
// .env.mobile — the exact host/port/path depends on your serve topology (the ACL
// caps homelab ports and 443 already serves Umami), so it is NOT hardcoded here.
// NO LITERAL DSN (operator, 2026-10-03). This line used to end in
// `|| 'http://<key>@homelab:8090/8'`, which is the same mistake the Umami block carried: a dev
// target baked into the source, pointing at a tailnet hostname that does not resolve from every
// account, with a project id nothing verifies. The Umami twin turned out to reference a website
// that does not exist at all — every dev event came back "Website not found." — and a hardcoded
// DSN fails the same way, silently, because Sentry's transport swallows its own errors too.
//
// Both tiers now come from the environment and nothing else:
//
//   VITE_SENTRY_DSN_PLAYER      — prod, baked as a docker build-arg
//   VITE_SENTRY_DSN_PLAYER_DEV  — the dev rung; `.env.mobile` supplies an https URL for on-device
//                                 use, because a native WebView blocks a plain-http DSN as mixed
//                                 content
//
// With neither set, error reporting is a true no-op. That is the right failure: a build that
// forgets its DSN reports nothing, rather than posting into someone else's project.
const DEV_SENTRY_DSN_PLAYER = (import.meta.env.VITE_SENTRY_DSN_PLAYER_DEV as string) || ''
const devDefault = import.meta.env.DEV && import.meta.env.VITE_ANALYTICS_OFF !== '1'
// Native dev↔prod switch (#1310): when the shell's tier is 'dev', errors go to the tailnet
// player-dev GlitchTip + environment='dev' — same channel split as the web dev rung. prod (+ web +
// release) keeps the baked prod DSN. Umami stays the prod site (unified UX), so it is NOT switched.
const nativeDevTier = tierSwitchEnabled() && getTier() === 'dev'
const SENTRY_DSN_PLAYER = nativeDevTier
  ? DEV_SENTRY_DSN_PLAYER
  : import.meta.env.VITE_SENTRY_DSN_PLAYER || (devDefault ? DEV_SENTRY_DSN_PLAYER : '')
if (SENTRY_DSN_PLAYER) {
  // @sentry/capacitor wraps @sentry/vue: one init drives the JS SDK AND the native
  // sentry-cocoa / sentry-android SDKs, so a crash in the Swift/Kotlin shell (outside
  // the WebView) is captured too — not just WebView JS errors. On web the native layer
  // is a no-op. Vue-specific options live under siblingOptions.vueOptions; SentryVue.init
  // is forwarded as the 2nd arg. Core options (dsn/environment/release/…) stay top-level.
  Sentry.init(
    {
      dsn: SENTRY_DSN_PLAYER,
      environment: nativeDevTier ? 'dev' : import.meta.env.PROD ? 'prod' : 'dev',
      release: __BUILD_SHA__ || undefined,
      // Keep PII off by default.
      sendDefaultPii: false,
      // THE SEARCH TERM REACHES GLITCHTIP THROUGH NAVIGATION BREADCRUMBS (#2264). Umami's
      // `data-exclude-search` does nothing for this second sink, and `sendDefaultPii: false` does
      // not cover it either — that option governs IP address, cookies and user data, not query
      // strings. The measurement and the reasoning live in `services/telemetryScrub.ts`, where the
      // behaviour is unit-tested; an inline hook here could only ever be checked by grepping this
      // file for its own name.
      beforeBreadcrumb: scrubNavigationBreadcrumb,
      beforeSend: scrubEventRequestUrl,
      // Conservative tracing rate — parity with the viewer.
      tracesSampleRate: 0.1,
      // Tag every event so the player stream stays separable from api / pipeline
      // / viewer in the GlitchTip UI. `platform` (web|ios|android) separates the
      // native-shell builds from the web player in the same stream (#1310).
      initialScope: {
        tags: { component: 'player', platform: platform() },
      },
      siblingOptions: {
        vueOptions: {
          app,
          attachProps: true,
          // Hook Vue's errorHandler so component render/lifecycle errors are captured.
          attachErrorHandler: true,
        },
      },
    },
    // Forward the init method from @sentry/vue.
    SentryVue.init,
  )
}

// Umami analytics for the consumer player — cookieless page + route tracking, mirroring orrery.
//
// The injection itself (and the `data-exclude-search` attribute that keeps the search term out of
// tracked URLs) lives in `services/analytics.ts`, which is the SINGLE injection path. It moved out
// of this file when sign-out gained the need to replace the tracker (`resetIdentity`): two copies
// of the injection logic would have drifted the moment one of them gained an attribute and the
// other did not. Everything the old block explained — the dev rung over the tailnet, the
// build-arg-baked prod values, the native-https caveat for on-device dev analytics, and the
// fork-silent default when neither resolves — is documented there, next to the code that does it.
installUmami()

// Reopen where you left off after the OS ends the app in the background (#2278). Installed before
// `use(router)`, which starts the first navigation — the one the restore may redirect. The stores
// are reached lazily, inside the callbacks, because pinia is installed in the same chain below.
const lastPlaceDeps = {
  userId: () => useAuthStore().user?.user_id ?? null,
  nowPlaying: () => usePlayerStore().nowPlaying(),
  loadAt: (episode: SavedEpisode, seconds: number) => usePlayerStore().loadAt(episode, seconds),
  ensureAuthLoaded: () => useAuthStore().ensureLoaded(),
  // App.vue wires both again on mount; doing it here too is idempotent.
  prepareLocalSources: async (userId: string) => {
    usePlayerStore().setSourceResolver(localSourceFor)
    await useDownloadsStore().setNamespace(userId)
  },
}
installLastPlace(router, lastPlaceDeps)

app.use(createPinia()).use(router).use(i18n)

// How the app came back — warm, WebView reload or cold — once the first navigation (and so the
// restore) has settled (#2277). Going to the background also saves the place with its position.
void router.isReady().then(() => initLifecycle(() => recordPlace(router, lastPlaceDeps)))

// RFC-120 (#2009): route an EXPIRED session to the lure landing. A 401 fires this only when we
// still believe we're signed in (guards against a redirect loop — anonymous 401s are normal under
// login-first). Registered after pinia+router so the store and navigation are live.
let adjudicating401 = false
setOnUnauthorized(() => {
  const auth = useAuthStore()
  // `hasSession`, NOT `isAuthenticated`. Gating on the latter meant a device holding a DEAD native
  // token but no painted user fell straight through this handler: the token was never discarded,
  // no redirect happened, and the app sat on an admitted route with a signed-out masthead — every
  // authed read 401ing forever, across relaunches, with no self-heal. Reproduced on the simulator
  // 2026-09-16. An anonymous 401 (normal under login-first) still no-ops, because with no user and
  // no token `hasSession` is false.
  if (!auth.hasSession) return
  // ADJUDICATE the 401 — do not wipe on sight.
  //
  // This called `markSignedOut()`, which drops the token and the identity snapshot with NONE of
  // the guards `refresh()` carries. Since `api.ts` fires this handler on EVERY 401, the careful
  // truth table in `refresh()` — keep the token when the server rotated its key or is unwell,
  // keep the identity on a transport failure — was unreachable: the wipe always won the race.
  // The 2026-09-16 incident was hardened in one path and left open in the one that actually runs.
  //
  // `refresh()` re-asks `/me` and applies that truth table. The latch stops its own inner 401 from
  // re-entering here; the redirect happens only if it concludes the session is genuinely gone.
  if (adjudicating401) return
  adjudicating401 = true
  void auth
    .refresh()
    .finally(() => {
      adjudicating401 = false
      if (auth.hasSession) return
      const current = router.currentRoute.value
      if (current.name !== 'landing' && current.name !== 'login') {
        void router.replace({ name: 'landing', query: { redirect: current.fullPath } })
      }
    })
})

// Native prep that MUST land before the router's first navigation (which starts at mount):
//  - rehydrateNativeToken: sets the durable Bearer so the initial guard's GET /me is authenticated,
//    not an anonymous 401 that would wipe the device snapshot + content cache (advisor 2026-09-09);
//  - initGateCookie: seeds the cl_preview gate cookie so that first request also clears the
//    coming-soon gate.
// Both are no-ops on web/dev/release, so mount isn't meaningfully delayed there. Pinia is installed
// above, so the prefs IIFE below stays valid regardless of when mount lands.
void Promise.allSettled([rehydrateNativeToken(), initGateCookie()]).finally(() => app.mount('#app'))

// USERPREFS-1 (#1213) — hydrate the user preferences payload once at app
// init. Consumers (HomeView, PlayerView, future adopters) read via
// ``useUserPreferencesStore().get(key)`` and get server values when
// available, undefined when not. Fire-and-forget so mount doesn't wait
// on the network round-trip; consuming stores react when the promise
// resolves. Silent-degrade on 401 / offline.
void (async () => {
  const { useUserPreferencesStore } = await import('./stores/userPreferences')
  await useUserPreferencesStore().hydrate()
})()
