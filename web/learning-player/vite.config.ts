/// <reference types="vitest/config" />
import { execSync } from 'node:child_process'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { VitePWA } from 'vite-plugin-pwa'

/**
 * Build-time identity injected into the bundle so the running client
 * can identify itself. Critical for update-path debugging: without a
 * visible build version, "the PWA isn't updating" reports have no
 * evidence and it's easy to invent root causes (Guide §3).
 *
 * Prefers the git SHA (short) and ISO timestamp; falls back to
 * env-provided values so container builds without a .git can still
 * carry identity.
 */
function resolveBuildInfo(): { sha: string; time: string } {
  const envSha = process.env.BUILD_SHA
  if (envSha) return { sha: envSha.slice(0, 12), time: process.env.BUILD_TIME || new Date().toISOString() }
  try {
    const sha = execSync('git rev-parse --short=12 HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim()
    return { sha, time: new Date().toISOString() }
  } catch {
    return { sha: 'unknown', time: new Date().toISOString() }
  }
}

const BUILD_INFO = resolveBuildInfo()

// Deploy base — read from env so subpath deploys (e.g. behind a reverse
// proxy at /app/, or a preview under /pr-123/) can carry the correct
// SW scope + manifest paths + offline navigateFallback. Guide §5 war
// story: getting this wrong at root "works", getting it wrong under a
// base path silently 404s offline routing under only-that-base. Defaults
// to '/' so root deploys are unchanged.
const APP_BASE = (process.env.APP_BASE || '/').replace(/\/+$/, '') + '/'

// https://vite.dev/config/
export default defineConfig({
  base: APP_BASE,
  define: {
    __BUILD_SHA__: JSON.stringify(BUILD_INFO.sha),
    __BUILD_TIME__: JSON.stringify(BUILD_INFO.time),
  },
  plugins: [
    vue(),
    VitePWA({
      // "prompt" (vs "autoUpdate") gives users a visible "Reload to update"
      // toast instead of the silent-update-stall trap that hits users with
      // long-lived tabs. See src/composables/usePwaUpdate.ts + PwaUpdateToast.vue.
      registerType: 'prompt',
      manifest: {
        name: 'Learning Player',
        short_name: 'Learning',
        description:
          'A podcast learning player — transcript-synced playback over grounded intelligence.',
        theme_color: '#0E0D10',
        background_color: '#0E0D10',
        display: 'standalone',
        // Base-aware start_url + scope. At root APP_BASE='/'; under a
        // subpath deploy set APP_BASE=/app/ (etc.) and the manifest,
        // SW scope, and navigateFallback all line up. Guide §1 + §5.
        start_url: APP_BASE,
        scope: APP_BASE,
        icons: [
          // "any" purpose covers regular home-screen icons and the install prompt.
          { src: `${APP_BASE}icon-192.png`, sizes: '192x192', type: 'image/png', purpose: 'any' },
          { src: `${APP_BASE}icon-512.png`, sizes: '512x512', type: 'image/png', purpose: 'any' },
          // "maskable" purpose — Android crops the icon into a themed shape; without
          // a maskable icon the OS wraps the icon in a white rounded box. Safe-zone
          // is the central ~80 % (outer ~10 % may be clipped).
          {
            src: `${APP_BASE}maskable-512.png`,
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable',
          },
        ],
      },
      workbox: {
        // Cache the app shell + GET API responses (stale-while-revalidate).
        // Audio is NEVER cached/proxied by the SW (bridge-never-rehost): origin
        // media URLs are excluded from runtime caching.
        // Base-aware: offline deep-links route back to the app shell under
        // the deployed base path, not always root.
        navigateFallback: `${APP_BASE}index.html`,
        // Precache the SHELL ONLY (js/css/fonts/small icons). Never precache
        // large media — that's the guide's #1 shipping trap. Artwork + audio
        // stay on the runtime path.
        globPatterns: ['**/*.{js,css,html,woff2,ico,svg}'],
        globIgnores: ['**/artwork/**', '**/audio/**'],
        // Hard per-entry cap — belt-and-suspenders against a large asset
        // accidentally ending up in dist/ and inflating the precache.
        maximumFileSizeToCacheInBytes: 2 * 1024 * 1024,
        // Purge previous-version precaches on activate — without this they
        // accumulate on every SW version bump and eventually get evicted
        // (aggressively on iOS).
        cleanupOutdatedCaches: true,
        runtimeCaching: [
          {
            // Artwork is content-addressed + served immutable → cache hard, keep on-device.
            urlPattern: ({ url }: { url: URL }) => url.pathname === '/api/app/artwork',
            handler: 'CacheFirst',
            options: {
              cacheName: 'app-artwork',
              expiration: { maxEntries: 500, maxAgeSeconds: 60 * 60 * 24 * 30 },
            },
          },
          {
            // Shared, non-user GET reads (catalog/episode/segments/search) — SWR is safe.
            // Per-user + auth endpoints (/me, /queue, /playback, /auth) are EXCLUDED: caching
            // them risks serving one session's state to another across sign-in/out.
            urlPattern: ({ url }: { url: URL }) =>
              url.pathname.startsWith('/api/app/') &&
              !/^\/api\/app\/(me|queue|playback|auth)\b/.test(url.pathname),
            handler: 'StaleWhileRevalidate',
            options: {
              cacheName: 'api-app',
              // Guide §2: always bound runtime caches or storage grows
              // until the browser evicts the whole SW (iOS punishes this
              // hardest). 200 responses × 7 days is generous for shared
              // catalog/episode data.
              expiration: {
                maxEntries: 200,
                maxAgeSeconds: 60 * 60 * 24 * 7,
              },
            },
          },
        ],
      },
    }),
  ],
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000',
        // changeOrigin:false preserves the Host (localhost:5174) so the API builds
        // SAME-ORIGIN OAuth callback URLs — otherwise the mock/Google sign-in redirect
        // lands on the API origin and the session cookie is set on the wrong host.
        changeOrigin: false,
      },
    },
  },
  // Same proxy for `vite preview` — used by the full-stack e2e (real API, no mocks).
  // changeOrigin:false preserves the Host so the API builds same-origin OAuth callback URLs
  // (so the mock-provider sign-in flow + session cookie work on the preview origin).
  preview: {
    port: 4174,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8011',
        changeOrigin: false,
      },
    },
  },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'happy-dom',
    coverage: {
      provider: 'v8',
      reporter: ['text-summary', 'lcov', 'json-summary'],
      // Ratchetable floor — raise as the component/store/test surface grows.
      thresholds: {
        statements: 60,
        branches: 60,
        functions: 60,
        lines: 60,
      },
    },
  },
})
