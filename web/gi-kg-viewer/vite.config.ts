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
    // UI test-coverage track (#914): a parallel gate to the Python coverage gate.
    // Thresholds are a ratchetable floor set a few points below the current
    // baseline (stmts 77 / br 68 / fn 76 / ln 79). Raise them as coverage grows.
    // Default include = files exercised by tests (no `all`), matching that baseline.
    coverage: {
      provider: 'v8',
      reporter: ['text-summary', 'lcov', 'json-summary'],
      thresholds: {
        statements: 75,
        branches: 65,
        functions: 73,
        lines: 76,
      },
    },
  },
})
