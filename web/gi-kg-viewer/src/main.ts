import { createPinia } from 'pinia'
import * as Sentry from '@sentry/vue'
import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import { applyPreset } from './theme/theme'

applyPreset()

const app = createApp(App)

// Sentry init for the viewer — gated on ``VITE_SENTRY_DSN_VIEWER`` so
// the default behaviour (no DSN) stays a true no-op for dev / CI / any
// build that hasn't passed the build-arg through. The DSN reaches Vite
// at build time via ``VITE_*`` env vars; the docker viewer build needs
// to pass ``VITE_SENTRY_DSN_VIEWER`` as a build-arg.
//
// See RFC-081 §Layer 2 + issue #681. Pairs with the Python-side init
// in ``src/podcast_scraper/utils/sentry_init.py`` so api / pipeline /
// viewer all stream into separate Sentry projects with
// ``component=`` tags on each event.
const SENTRY_DSN_VIEWER = import.meta.env.VITE_SENTRY_DSN_VIEWER as string | undefined
if (SENTRY_DSN_VIEWER) {
  Sentry.init({
    app,
    dsn: SENTRY_DSN_VIEWER,
    environment: (import.meta.env.VITE_PODCAST_ENV as string) || 'dev',
    release: (import.meta.env.VITE_PODCAST_RELEASE as string) || undefined,
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
app.mount('#app')
