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
    // baseline. Ratcheted after the graph-UI + app-wide test push (API/stores/
    // utils to ~100%, graph stores/utils + component mount tests): baseline is now
    // stmts 84.9 / br 74.3 / fn 89.0 / ln 86.5. Raise them as coverage grows.
    // Default include = files exercised by tests (no `all`), matching that baseline.
    coverage: {
      provider: 'v8',
      reporter: ['text-summary', 'lcov', 'json-summary'],
      thresholds: {
        statements: 82,
        branches: 71,
        functions: 86,
        lines: 84,
      },
    },
  },
})
