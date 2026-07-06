import { defineConfig, devices } from '@playwright/test'

/**
 * E2E config for the consumer Learning Player. Boots the Vite preview server and runs the
 * smoke + (later) full listen→capture specs. Mobile-first: the default project emulates a
 * phone viewport, matching the app's primary target (UXS-011).
 */
export default defineConfig({
  testDir: './e2e',
  // e2e/validation/** is Tier-3 — runs under playwright.validation.config.ts
  // against a separately-booted `make serve-for-validation` stack, NOT the
  // fast-tier preview here. Excluding it prevents these specs from firing
  // twice + failing because the shared serviceWorkers:'block' default is
  // wrong for the SW-driven validation walks.
  testIgnore: ['**/validation/**'],
  fullyParallel: true,
  // The heavy auth-gated specs (capture, consolidation) sign in as ISOLATED per-(spec,project) mock
  // identities (see e2e/helpers.ts) so they never share per-user files — eliminating the
  // concurrency race at its source. auth-queue keeps the default mock user via the real Sign-in UI
  // (its own 2-project scenario, reliable). No globalSetup / retry band-aid needed.
  retries: process.env.CI ? 2 : 0,
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
      // Paths are relative to this config's cwd — web/learning-player/ —
      // so `../..` traverses back to the repo root (where .venv, src/,
      // and tests/ live). Missed this on slice 14; caught by
      // ``make test-app-e2e`` locally when the first run of playwright
      // failed with `../.venv/bin/python: No such file or directory`.
      command:
        '../../.venv/bin/python -m podcast_scraper.cli serve ' +
        '--output-dir ../../tests/fixtures/app-validation-corpus/v3 --port 8011 --host 127.0.0.1',
      url: 'http://127.0.0.1:8011/api/health',
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
      env: {
        PYTHONPATH: '../../src',
        APP_OAUTH_PROVIDER: 'mock',
        APP_SESSION_SECRET: 'e2e-secret',
        // Allow the mock dev identity through the access policy (default is allowlist/deny).
        APP_SIGNUP_MODE: 'open',
        // Personalized discovery ON so the recommender A/B (recommendation.spec) can assert the feed
        // re-ranks toward a followed interest. With no interests the feed is recency (unchanged), so
        // this is inert for every other spec.
        APP_PERSONALIZED_RANKING: 'true',
        // Keep per-user writes (queue/profile/interests) OUT of the committed corpus tree.
        // Relative to the webServer cwd (web/learning-player/); the server resolve()s it against cwd.
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
