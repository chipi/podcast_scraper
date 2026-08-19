import { createApp } from 'vue'
import { createPinia } from 'pinia'
import * as Sentry from '@sentry/capacitor'
import * as SentryVue from '@sentry/vue'
import './style.css'
import App from './App.vue'
import { router } from './router'
import { i18n } from './i18n'
import { applyTheme } from './theme/theme'
import { initGateCookie, platform } from './services/native'
import { getTier, tierSwitchEnabled } from './services/tier'

applyTheme('dark')

// Expose build identity for update-path debugging. When a user reports
// 'the PWA isn't updating', the running client's sha + time can be read
// from window.__buildInfo (DevTools console or a support form) to
// distinguish 'stuck client' from 'cache never invalidated'. See
// vite.config.ts `define:` block for the injection.
window.__buildInfo = { sha: __BUILD_SHA__, time: __BUILD_TIME__ }

console.info(`[app] Learning Player build=${__BUILD_SHA__} time=${__BUILD_TIME__}`)

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
const DEV_SENTRY_DSN_PLAYER =
  import.meta.env.VITE_SENTRY_DSN_PLAYER_DEV || 'http://66dba2f7683848c8b4ef0968ff073e82@homelab:8090/8'
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

// Umami analytics for the consumer player — cookieless, privacy-friendly page +
// route tracking, mirroring orrery. Gated on both VITE_UMAMI_WEBSITE_ID and
// VITE_UMAMI_SRC (the public tracking-script URL on the analytics ingest edge,
// e.g. https://analytics.<domain>/script.js), baked at build time via docker
// build-args. Both empty by default => true no-op for dev / CI / any build
// without the args. Umami's script auto-tracks SPA route changes (it hooks the
// History API), so injecting the tag is all that's needed.
// Dev rung (see the Sentry block above): in `vite dev` analytics go to the
// dedicated `player-dev` Umami site via `homelab` (tailnet, no fixed IP). Prod
// overrides via the build-arg-baked VITE_UMAMI_* (public HTTPS analytics edge).
// Same native-https caveat as the dev DSN above: on-device the http script is
// mixed-content-blocked, so inject the tailscale-serve https URL via
// VITE_UMAMI_SRC_DEV in .env.mobile for dev-tier on-device analytics.
const DEV_UMAMI_SRC = import.meta.env.VITE_UMAMI_SRC_DEV || 'http://homelab:3001/script.js'
const DEV_UMAMI_WEBSITE_ID = '30384fd4-b22b-406c-b5f6-054a0e0d16d1'
const UMAMI_WEBSITE_ID =
  import.meta.env.VITE_UMAMI_WEBSITE_ID || (devDefault ? DEV_UMAMI_WEBSITE_ID : '')
const UMAMI_SRC = import.meta.env.VITE_UMAMI_SRC || (devDefault ? DEV_UMAMI_SRC : '')
if (UMAMI_WEBSITE_ID && UMAMI_SRC) {
  const umami = document.createElement('script')
  umami.defer = true
  umami.src = UMAMI_SRC
  umami.setAttribute('data-website-id', UMAMI_WEBSITE_ID)
  document.head.appendChild(umami)
}

app.use(createPinia()).use(router).use(i18n)
// Native prod tier: seed the cl_preview gate cookie into the native jar BEFORE mount, so the very
// first API call (incl. a returning signed-in user's rehydrated Bearer request) already clears the
// coming-soon gate via the cookie and doesn't 401. Resolves immediately (no-op) on web/dev/release,
// so mount isn't meaningfully delayed there. Pinia is installed above, so the prefs IIFE below stays
// valid regardless of when mount lands.
void initGateCookie().finally(() => app.mount('#app'))

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
