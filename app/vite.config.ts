/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { VitePWA } from 'vite-plugin-pwa'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    VitePWA({
      registerType: 'autoUpdate',
      manifest: {
        name: 'Learning Player',
        short_name: 'Learning',
        description:
          'A podcast learning player — transcript-synced playback over grounded intelligence.',
        theme_color: '#0E0D10',
        background_color: '#0E0D10',
        display: 'standalone',
        start_url: '/',
        icons: [
          { src: '/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icon-512.png', sizes: '512x512', type: 'image/png' },
        ],
      },
      workbox: {
        // Cache the app shell + GET API responses (stale-while-revalidate).
        // Audio is NEVER cached/proxied by the SW (bridge-never-rehost): origin
        // media URLs are excluded from runtime caching.
        navigateFallback: '/index.html',
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
            options: { cacheName: 'api-app' },
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
        changeOrigin: true,
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
