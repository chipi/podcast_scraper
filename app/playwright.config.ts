import { defineConfig, devices } from '@playwright/test'

/**
 * E2E config for the consumer Learning Player. Boots the Vite preview server and runs the
 * smoke + (later) full listen→capture specs. Mobile-first: the default project emulates a
 * phone viewport, matching the app's primary target (UXS-011).
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  // All specs share one mock-OAuth identity (the provider returns a fixed subject) and run
  // concurrently. The per-user store is deterministic + atomic, but a COLD first run (servers +
  // app JIT still warming) can momentarily slow an auth-gated assertion; CI absorbs that with the
  // retries below, and the 10s expect tolerance (up from 5s) covers heavy-parallel-load slowness.
  // A warm run is reliably green. (A globalSetup warm-up was tried + dropped — the cold OAuth
  // redirect was itself the slow path it was meant to fix.)
  retries: process.env.CI ? 2 : 1,
  expect: { timeout: 10_000 },
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: 'http://127.0.0.1:4174',
    trace: 'on-first-retry',
    // Block the PWA service worker so the e2e exercises the real network path deterministically
    // (the SW would otherwise intercept /api/app/* with stale-while-revalidate).
    serviceWorkers: 'block',
  },
  projects: [
    { name: 'mobile-chrome', use: { ...devices['Pixel 7'] } },
    { name: 'desktop-chrome', use: { ...devices['Desktop Chrome'] } },
  ],
  // Full-stack, NO MOCKS: the real consumer API serves a COMMITTED, deterministically-synthesized
  // corpus, and the built app is proxied to it (same-origin via preview proxy). This catches
  // server-contract bugs (e.g. the transcript_file_path metadata key) that a client-mocked e2e
  // cannot.
  //
  // The corpus is tests/fixtures/app-validation-corpus/v2 — checked in, version-pinned, and built
  // by scripts/build_app_validation_corpus.py (deterministic, no pipeline, no ML). There is NO
  // build step here: `serve` reads the committed corpus directly, so boot is fast and stable.
  // Per-user runtime state (queue/profile/interests the API writes) is redirected via APP_DATA_DIR
  // to a gitignored ephemeral dir so the committed corpus tree is never mutated.
  webServer: [
    {
      command:
        '../.venv/bin/python -m podcast_scraper.cli serve ' +
        '--output-dir ../tests/fixtures/app-validation-corpus/v2 --port 8011 --host 127.0.0.1',
      url: 'http://127.0.0.1:8011/api/health',
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
      env: {
        PYTHONPATH: '../src',
        APP_OAUTH_PROVIDER: 'mock',
        APP_SESSION_SECRET: 'e2e-secret',
        // Allow the mock dev identity through the access policy (default is allowlist/deny).
        APP_SIGNUP_MODE: 'open',
        // Keep per-user writes (queue/profile/interests) OUT of the committed corpus tree.
        // Relative to the webServer cwd (app/); the server resolve()s it against cwd.
        APP_DATA_DIR: 'e2e/.app-state',
      },
    },
    {
      command: 'npm run build && npm run preview -- --port 4174 --strictPort --host 127.0.0.1',
      url: 'http://127.0.0.1:4174',
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: { VITE_API_TARGET: 'http://127.0.0.1:8011' },
    },
  ],
})
