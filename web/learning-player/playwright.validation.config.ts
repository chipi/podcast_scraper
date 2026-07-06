import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { defineConfig, devices } from '@playwright/test'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/**
 * Consumer Learning Player — Tier-3 real-corpus validation config.
 *
 * Mirrors the viewer's Tier-3 story (see
 * `web/gi-kg-viewer/playwright.validation.config.ts`): drives a real
 * browser against a running `make serve-for-validation` stack (API on
 * :8000, app served here on :5175) with a real corpus on disk, walking
 * every player-critical surface and capturing named screenshots for
 * post-hoc inspection.
 *
 * Separate from `playwright.config.ts` (the regular fast-e2e config which
 * boots its own preview + committed fixture corpus at :4174 and runs
 * 12 broad-surface specs). Tier-3 is:
 *   - **sequential** (`workers=1`) so screenshots + logs are deterministic
 *   - **screenshotted on every step** — the artifact IS the value
 *   - **corpus-swappable** via `APP_CORPUS_PATH` env — operator-driven
 *     runs against production-shape data (nightly CI defaults to the
 *     committed synthetic corpus)
 *   - **no built-in webServer** — expects `make serve-for-validation`
 *     already running (matches viewer Tier-3 pattern)
 *
 * Run:
 *   make serve-for-validation  # in another terminal
 *   cd app
 *   APP_CORPUS_PATH=/abs/path/to/your/corpus \
 *     node_modules/.bin/playwright test --config playwright.validation.config.ts
 *
 * Screenshots land in `app/validation-results/`.
 */
export default defineConfig({
  testDir: './e2e/validation',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  timeout: 120_000,
  expect: { timeout: 20_000 },
  outputDir: path.join(__dirname, 'validation-results'),
  use: {
    baseURL: 'http://localhost:5175',
    trace: 'retain-on-failure',
    screenshot: 'on',
    // Real service worker install path — matches production install shape.
    serviceWorkers: 'allow',
    ...devices['Desktop Chrome'],
    viewport: { width: 1440, height: 900 },
  },
  projects: [{ name: 'chromium', use: {} }],
})
