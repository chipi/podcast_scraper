/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const viewerBuildTimestamp = new Date().toISOString()

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  define: {
    __VIEWER_BUILD_TIMESTAMP__: JSON.stringify(viewerBuildTimestamp),
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    include: ['src/**/*.test.ts'],
    // Ensure Vitest never initializes PostHog (no token); avoids flaky network side effects.
    env: {
      VITE_POSTHOG_PROJECT_TOKEN: '',
    },
  },
})
