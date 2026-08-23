import { createPinia } from 'pinia'
import * as Sentry from '@sentry/vue'
import { createApp } from 'vue'
/**
 * Self-hosted webfonts (#1619). These used to be `<link>`s to fonts.googleapis.com in
 * `index.html`, which made the app depend on a third-party CDN at runtime: with no internet there
 * were no fonts, every operator's browser announced itself to Google on each load, and the render
 * was blocked on a cross-origin round trip. In the offline e2e suite a slow CDN surfaced as
 * `downloadable font: download failed` on the console, failing every handoff row that asserts zero
 * console errors.
 *
 * Bundled via @fontsource, so the woff2 files ship with the app and the exact font version is
 * pinned in the lockfile — Google republishing a face cannot silently change our typography.
 * Latin subsets only, and only the weights `style.css` asks for, matching the old query string:
 * Inter 400/500/600/700 and JetBrains Mono 400/500. Both are OFL, so redistribution is fine.
 */
import '@fontsource/inter/latin-400.css'
import '@fontsource/inter/latin-500.css'
import '@fontsource/inter/latin-600.css'
import '@fontsource/inter/latin-700.css'
import '@fontsource/jetbrains-mono/latin-400.css'
import '@fontsource/jetbrains-mono/latin-500.css'
import './style.css'
import App from './App.vue'
import { applyPreset } from './theme/theme'
import { initAnalytics } from './lib/analytics'

applyPreset()

const app = createApp(App)

// Sentry init for the viewer — gated on ``VITE_SENTRY_DSN_VIEWER`` so
// the default behaviour (no DSN) stays a true no-op for dev / CI / any
// build that hasn't passed the build-arg through. The DSN reaches Vite
// at build time via ``VITE_*`` env vars; the docker viewer build needs
// to pass ``VITE_SENTRY_DSN_VIEWER`` as a build-arg.
//
// ``environment`` is RUNTIME-injected via nginx ``sub_filter`` (Item 1 /
// RFC-082 follow-up) so ONE viewer image serves prod + preprod with the
// right Sentry env tag. nginx rewrites the served index.html on its way
// out, prepending a ``<script>`` that sets ``window.__PODCAST_ENV__``,
// ``window.__PODCAST_RELEASE__``, and optionally
// ``window.__PODCAST_DEFAULT_CORPUS_PATH__``. See
// docker/viewer/default.conf.template.
//
// See RFC-081 §Layer 2 + issue #681. Pairs with the Python-side init
// in ``src/podcast_scraper/utils/sentry_init.py`` so api / pipeline /
// viewer all stream into separate Sentry projects with
// ``component=`` tags on each event.
//
// Dev rung of the env ladder (dev → prod; the operator has no staging). In
// ``vite dev``, with no build-injected DSN, errors go to the dedicated
// ``operator-dev`` GlitchTip project via the Tailscale host ``homelab`` — NO
// fixed IP, so only a device on the tailnet resolves it; a stranger who runs
// the repo reports nothing (the transport silently fails). The key is a public
// browser id (ships in the bundle) — safe to commit. ``VITE_ANALYTICS_OFF=1``
// (set by the vitest + playwright configs) suppresses the dev default in tests.
const DEV_SENTRY_DSN_VIEWER = 'http://53a88592c99e48bc8d505d258597ab78@homelab:8090/9'
const devSentryDefault =
  import.meta.env.DEV && import.meta.env.VITE_ANALYTICS_OFF !== '1'
    ? DEV_SENTRY_DSN_VIEWER
    : undefined
const SENTRY_DSN_VIEWER =
  (import.meta.env.VITE_SENTRY_DSN_VIEWER as string | undefined) || devSentryDefault
if (SENTRY_DSN_VIEWER) {
  const w = window as Window & {
    __PODCAST_ENV__?: string
    __PODCAST_RELEASE__?: string
  }
  Sentry.init({
    app,
    dsn: SENTRY_DSN_VIEWER,
    environment: w.__PODCAST_ENV__ || (import.meta.env.DEV ? 'dev' : 'prod'),
    release: w.__PODCAST_RELEASE__ || (import.meta.env.VITE_PODCAST_RELEASE as string) || undefined,
    // Keep PII off by default.
    sendDefaultPii: false,
    // Conservative tracing rate — viewer is bursty (graph re-renders,
    // explore queries) and the free tier has 10k transactions/mo.
    tracesSampleRate: 0.1,
    // Tag every event so the api / pipeline / viewer streams stay
    // separable in the Sentry UI.
    initialScope: {
      tags: { component: 'viewer' },
    },
  })
}

app.use(createPinia())

// USERPREFS-1 — kick off cross-device preference hydration before mount so
// stores that check `useUserPreferencesStore().get(...)` at init time see
// the server-supplied values (falls through to localStorage / local defaults
// silently when the user isn't authenticated or the network is unavailable).
// Fire-and-forget: mount doesn't wait on the hydrate to avoid delaying first
// paint; consuming stores react when the promise resolves.
void (async () => {
  const { useUserPreferencesStore } = await import('./stores/userPreferences')
  await useUserPreferencesStore().hydrate()
})()

// Umami analytics — cookieless page + custom-event tracking (replaces PostHog
// Cloud, 2026-07-24). Injects the tracking script once when enabled; a no-op
// for fork / non-dev builds with no env. Vue errors are captured by the
// @sentry/vue integration installed by Sentry.init({ app }) above, so no
// custom errorHandler wrapper is needed anymore.
initAnalytics()

app.mount('#app')
