import { createApp } from 'vue'
import { createPinia } from 'pinia'
import * as Sentry from '@sentry/vue'
import './style.css'
import App from './App.vue'
import { router } from './router'
import { i18n } from './i18n'
import { applyTheme } from './theme/theme'

applyTheme('dark')

// Expose build identity for update-path debugging. When a user reports
// "the PWA isn't updating", the running client's sha + time can be read
// from window.__buildInfo (DevTools console or a support form) to
// distinguish "stuck client" from "cache never invalidated". See
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
const SENTRY_DSN_PLAYER = import.meta.env.VITE_SENTRY_DSN_PLAYER
if (SENTRY_DSN_PLAYER) {
  Sentry.init({
    app,
    dsn: SENTRY_DSN_PLAYER,
    environment: import.meta.env.PROD ? 'prod' : 'dev',
    release: __BUILD_SHA__ || undefined,
    // Keep PII off by default.
    sendDefaultPii: false,
    // Conservative tracing rate — parity with the viewer.
    tracesSampleRate: 0.1,
    // Tag every event so the player stream stays separable from api / pipeline
    // / viewer in the GlitchTip UI.
    initialScope: {
      tags: { component: 'player' },
    },
  })
}

app.use(createPinia()).use(router).use(i18n).mount('#app')

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
