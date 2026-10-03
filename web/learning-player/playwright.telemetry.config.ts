import { defineConfig, devices } from '@playwright/test'

/**
 * Tier-4: the analytics arc's end-to-end proof (#2263, epic slices 1–5).
 *
 * WHY THIS IS A SEPARATE CONFIG, not specs under `playwright.config.ts`
 * ────────────────────────────────────────────────────────────────────
 * Everything else in the suite runs against `npm run preview`, which is a PRODUCTION build. For a
 * production build `import.meta.env.DEV` is false, so `main.ts` deliberately refuses the dev
 * telemetry rung: the Sentry DSN resolves to `VITE_SENTRY_DSN_PLAYER || ''` and the dev DSN is
 * never consulted. That is correct behaviour — a release build must not fall back to a dev target —
 * but it means a preview build cannot exercise the dev rung at all.
 *
 * So this config serves the app with `vite dev` instead. That is the real web dev tier: the dev
 * GlitchTip project is used, `environment` resolves to 'dev', and the Umami site is the dev site.
 * Proving the rung the way it actually ships beats proving a rung assembled for the test.
 *
 * The API and mock-audio servers are the SAME ones the main config boots — the real consumer API
 * over the committed validation corpus, no client mocks — because the events under test carry
 * properties derived from real server responses (`from_followed_show`, result ranks, entity kinds).
 * A mocked API would let a wrong property pass.
 *
 * WHAT IT TALKS TO
 * ────────────────
 * Umami  → the DEV website `3ccaa1bc-…` ("Player (dev)") on 127.0.0.1:3001.
 * GlitchTip → the DEV project `player-dev` (id 20) on 127.0.0.1:8090.
 *
 * Neither is the deployed target. Prod Umami site `cd384a3e-…` holds 562 real events and prod
 * GlitchTip project 5 takes the deployed player's errors; nothing here touches either. The dev
 * targets exist precisely so this run cannot contaminate live data.
 *
 * Run: `npx playwright test -c playwright.telemetry.config.ts`
 */
export default defineConfig({
  testDir: './e2e/telemetry',
  globalSetup: './e2e/telemetry/globalSetup.ts',
  fullyParallel: false, // these specs assert on a shared external sink; serialise them
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 20_000 },
  retries: 0, // a retry would double-count events in Umami and make the surface numbers lie
  reporter: [['list'], ['json', { outputFile: 'e2e/telemetry/.report.json' }]],
  use: {
    baseURL: 'http://127.0.0.1:4199',
    trace: 'retain-on-failure',
    // Block the PWA service worker: it would intercept /api/app/* with stale-while-revalidate and
    // make the server-derived event properties non-deterministic.
    serviceWorkers: 'block',
    // Umami's collector drops obvious bot user-agents and answers {"beep":"boop"} — a 200 that
    // looks like success and stores nothing. Playwright's default UA is one of those, which is why
    // an earlier probe appeared to work and wrote no rows. A realistic desktop Chrome UA is the
    // difference between a real measurement and a green test over an empty table.
    userAgent:
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
  },
  projects: [{ name: 'desktop-chrome', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command: '../../.venv/bin/python ../../scripts/tools/run_e2e_mock_server.py --port 18765',
      url: 'http://127.0.0.1:18765/audio/p05_e03.mp3',
      reuseExistingServer: true,
      timeout: 120_000,
      env: { PYTHONPATH: '../../src:../..' },
    },
    {
      command:
        'node e2e/prepare-corpus.mjs && ../../.venv/bin/python -m podcast_scraper.cli serve ' +
        '--output-dir .e2e-corpus/v3 --port 8011 --host 127.0.0.1',
      url: 'http://127.0.0.1:8011/api/health',
      reuseExistingServer: true,
      timeout: 180_000,
      env: {
        PYTHONPATH: '../../src',
        HF_HUB_OFFLINE: '1',
        TRANSFORMERS_OFFLINE: '1',
        HF_HOME: process.env.HF_HOME || `${process.env.HOME}/.cache/huggingface`,
        HF_HUB_CACHE: process.env.HF_HUB_CACHE || `${process.env.HOME}/.cache/huggingface/hub`,
        APP_OAUTH_PROVIDER: 'mock',
        APP_SESSION_SECRET: 'e2e-secret',
        APP_SIGNUP_MODE: 'open',
        APP_PERSONALIZED_RANKING: 'true',
        APP_TRENDING_NOW: '2026-07-20T00:00:00Z',
        APP_MOMENTUM_MIN_TOTAL: '1',
        APP_DATA_DIR: 'e2e/.telemetry-state',
      },
    },
    {
      // `vite dev`, not preview — see the header. 4199 keeps it off 5174/4174 so a developer's own
      // dev server or the main e2e preview can stay up alongside this run.
      command: 'npm run dev -- --port 4199 --strictPort --host 127.0.0.1',
      url: 'http://127.0.0.1:4199',
      reuseExistingServer: true,
      timeout: 180_000,
      env: { VITE_API_TARGET: 'http://127.0.0.1:8011' },
    },
  ],
})
