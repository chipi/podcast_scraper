import { defineConfig, devices } from '@playwright/test'

/**
 * POST-DEPLOY LIVE SMOKE (#43) — runs against the DEPLOYED operator viewer at
 * operator.closelistening.app, NOT a local build. There is NO webServer: it targets the
 * live origin directly. Mirrors the player's playwright.live.config.ts.
 *
 * The pre-launch coming-soon gate is passed with `httpCredentials` (the /preview basic-auth,
 * the SAME marko/guest doorman as the player). Playwright attaches that auth to EVERY request
 * — navigations included — so the SPA + its /api calls all clear the gate; one spec still
 * asserts the public (no creds) sees coming-soon.
 *
 * Unlike the player, the operator viewer is a LOGIN WALL behind the gate: with a real provider
 * (google on prod) nothing renders until a ≥creator Google session exists, which can't be
 * completed headless. So the gated specs stop at the sign-in entrypoint and assert it 307s to
 * Google with the correct HTTPS callback redirect_uri (the exact thing the 2026-07-25
 * redirect_uri_mismatch broke). The deep authed viewer flow lives in stack-test (mock provider).
 *
 * Env:
 *   LIVE_BASE_URL          default https://operator.closelistening.app
 *   OPERATOR_PREVIEW_USER  gate basic-auth user (default: marko)
 *   OPERATOR_PREVIEW_PASS  gate basic-auth password (REQUIRED — falls back to PLAYER_PREVIEW_PASS,
 *                          the identical doorman; skips gated specs if neither is set)
 *
 * Run:  OPERATOR_PREVIEW_PASS='…' npm run test:e2e:live
 */
const baseURL = process.env.LIVE_BASE_URL || 'https://operator.closelistening.app'
const username = process.env.OPERATOR_PREVIEW_USER || 'marko'
// The operator doorman reuses the player's basic-auth hashes, so PLAYER_PREVIEW_PASS works too.
const password = process.env.OPERATOR_PREVIEW_PASS || process.env.PLAYER_PREVIEW_PASS || ''

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
    // Undefined when no password is provided so the gated specs skip cleanly instead of
    // hammering the gate with empty creds. `origin`-scoped so the basic-auth is NOT sent to
    // cross-origin destinations (e.g. accounts.google.com during the OAuth redirect).
    httpCredentials: password ? { username, password, origin: baseURL } : undefined,
    trace: 'on-first-retry',
    // Block the PWA service worker so the smoke exercises the real network path.
    serviceWorkers: 'block',
  },
  projects: [{ name: 'desktop-chrome', use: { ...devices['Desktop Chrome'] } }],
})
