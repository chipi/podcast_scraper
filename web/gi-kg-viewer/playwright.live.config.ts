import { defineConfig, devices } from '@playwright/test'

/**
 * POST-DEPLOY LIVE SMOKE (#43) — runs against the DEPLOYED operator host at
 * operator.closelistening.app, NOT a local build. There is NO webServer: it targets the
 * live origin directly.
 *
 * The operator host is CLOSED (operator decision 2026-10-05: the operator surface is
 * tailnet-only), so the smoke asserts exactly that — every path answers the coming-soon page,
 * with or without the old preview credentials — and needs no secrets. The gated sign-in and
 * creator-account specs that ran here while the host was open were removed with the gate.
 *
 * Env:
 *   LIVE_BASE_URL  default https://operator.closelistening.app
 *
 * Run:  npm run test:e2e:live
 */
const baseURL = process.env.LIVE_BASE_URL || 'https://operator.closelistening.app'

export default defineConfig({
  testDir: './e2e/live',
  fullyParallel: false,
  // Live network — allow a couple retries for transient blips, but keep it snappy.
  retries: 2,
  reporter: process.env.CI ? 'github' : 'list',
  timeout: 45_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL,
    trace: 'on-first-retry',
    // Block the PWA service worker so the smoke exercises the real network path.
    serviceWorkers: 'block',
  },
  projects: [{ name: 'desktop-chrome', use: { ...devices['Desktop Chrome'] } }],
})
