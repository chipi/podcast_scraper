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
        // Override with VITE_API_TARGET when the API runs on a non-default port
        // (e.g. 8000 taken by another local service). Default unchanged.
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000',
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
    // baseline. Set after the graph-UI + app-wide test push (API/stores/utils to
    // ~100%, plus @vue/test-utils mount tests across graph + non-graph components).
    // NOTE on the moving denominator: v8 coverage scopes to files *imported by a
    // test* (no `all`). Mounting container components (shell, panels) pulls their
    // whole transitive .vue import tree into the denominator, so adding component
    // tests *broadens* scope and lowers the headline % even as absolute coverage
    // grows — the figure is now honest about the full component tree. Baseline at
    // this scope: stmts 81.1 / br 69.4 / fn 80.5 / ln 82.8. Raise as coverage grows.
    coverage: {
      provider: 'v8',
      reporter: ['text-summary', 'lcov', 'json-summary'],
      thresholds: {
        statements: 78,
        branches: 66,
        functions: 77,
        lines: 80,
      },
    },
  },
})
