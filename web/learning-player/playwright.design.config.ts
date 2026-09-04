import { defineConfig, devices } from '@playwright/test'

/**
 * Design-iteration harness (#1944) — screenshots for the critic loop, in seconds.
 *
 * ## Why this exists separately from the other three configs
 *
 * `playwright.config.ts` runs `npm run build && npm run preview`. That is correct for e2e —
 * you want the suite testing the bundle that ships. It is fatal for DESIGN work, where the whole
 * point is many cheap looks: every tweak pays a full production build before you see a pixel.
 *
 * The runbook claimed "<2 minutes per iteration". Measured, the stack boot alone is ~80s and a
 * preview rebuild adds ~30s on top. At that price divergence gets quietly truncated to whatever is
 * affordable, which defeats the exercise — you end up judging three directions because eight was
 * too slow, and calling it exploration.
 *
 * This config runs `vite dev` instead. HMR applies a CSS or template change in well under a
 * second, so a re-screenshot is the cost of the screenshot itself.
 *
 * The API and media servers are IDENTICAL to `playwright.config.ts` (same fixture corpus, same
 * momentum floor, same pinned trending date) so what the critic sees is the real app with real
 * data, not a mock. Only the frontend delivery changes.
 *
 * ## Not a replacement for the validation harness
 *
 * `playwright.validation.config.ts` (Tier-3) stays the tool for VALIDATION — real corpus, nightly
 * CI, screenshots as inspection artifacts. This is a separate, faster path for EXPLORATION. Do not
 * wire this into CI; it proves nothing about correctness.
 *
 * ## Viewport
 *
 * Pixel 7 only. The runbook's do-not-break list names mobile-first as non-negotiable and the app
 * is judged at 375px — so the critic should never be shown a desktop composition that flatters a
 * layout nobody uses.
 *
 * Run:
 *   npm run design:shots            # screenshot every surface
 *   npm run design:shots -- --grep home
 */
export default defineConfig({
  testDir: './e2e/design',
  fullyParallel: false,
  workers: 1,
  // No retries: a design screenshot that needed a retry is a screenshot of a different moment,
  // and the critic would be judging a race rather than a design.
  retries: 0,
  reporter: [['list']],
  timeout: 60_000,
  expect: { timeout: 15_000 },
  outputDir: 'design-results/.playwright',
  use: {
    baseURL: 'http://127.0.0.1:5174',
    ...devices['Pixel 7'],
    // Screenshots are the product here, so take them deliberately in the spec — not on failure.
    screenshot: 'off',
    trace: 'off',
  },
  projects: [{ name: 'pixel7', use: {} }],

  webServer: [
    {
      // Real fixture audio, so the player surface shows a transport rather than an error panel.
      command: '../../.venv/bin/python ../../scripts/tools/run_e2e_mock_server.py --port 18765',
      url: 'http://127.0.0.1:18765/audio/p05_e03.mp3',
      reuseExistingServer: true,
      timeout: 120_000,
      env: { PYTHONPATH: '../../src:../..' },
    },
    {
      // Identical to playwright.config.ts's api — same corpus, same flags. If these drift, the
      // critic starts judging a differently-populated app than the e2e suite tests.
      command:
        '../../.venv/bin/python -m podcast_scraper.cli serve ' +
        '--output-dir ../../tests/fixtures/app-validation-corpus/v3 --port 8011 --host 127.0.0.1',
      url: 'http://127.0.0.1:8011/api/health',
      reuseExistingServer: true,
      timeout: 120_000,
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
        // Pinned so trending is deterministic — a design shot must not change because time passed.
        APP_TRENDING_NOW: '2026-07-20T00:00:00Z',
        // The fixture is 36 episodes; without this the trending rails render empty and the critic
        // would be judging a half-populated Home.
        APP_MOMENTUM_MIN_TOTAL: '1',
        APP_DATA_DIR: 'e2e/.design-state',
      },
    },
    {
      // The whole point: dev server, not a production build.
      command: 'npm run dev -- --port 5174 --strictPort --host 127.0.0.1',
      url: 'http://127.0.0.1:5174',
      reuseExistingServer: true,
      timeout: 120_000,
      env: {
        VITE_API_TARGET: 'http://127.0.0.1:8011',
        VITE_MEDIA_TARGET: 'http://127.0.0.1:18765',
      },
    },
  ],
})
