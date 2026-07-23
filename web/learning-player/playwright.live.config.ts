import { defineConfig, devices } from '@playwright/test'

/**
 * POST-DEPLOY LIVE SMOKE (#43) — runs against the DEPLOYED player at closelistening.app,
 * NOT a local build. There is NO webServer: it targets the live origin directly.
 *
 * The pre-launch coming-soon gate is passed with `httpCredentials` (the /preview basic-auth).
 * Playwright attaches that auth to EVERY request — navigations included — so the app is
 * reachable regardless of the browser cookie/SW quirks the cookie gate is designed around;
 * a smoke test wants to prove the app works, not re-exercise the gate mechanics (one spec
 * still asserts the public sees coming-soon).
 *
 * Env:
 *   LIVE_BASE_URL        default https://closelistening.app
 *   PLAYER_PREVIEW_USER  gate basic-auth user (default: marko)
 *   PLAYER_PREVIEW_PASS  gate basic-auth password (REQUIRED — no default; skips gated specs if unset)
 *
 * Run:  PLAYER_PREVIEW_PASS='…' npm run test:e2e:live
 */
const baseURL = process.env.LIVE_BASE_URL || 'https://closelistening.app'
const username = process.env.PLAYER_PREVIEW_USER || 'marko'
const password = process.env.PLAYER_PREVIEW_PASS || ''

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
    // Block the PWA service worker so the smoke exercises the real network path (the SW would
    // otherwise intercept navigations + API calls with cached content — see the denylist fix).
    serviceWorkers: 'block',
  },
  projects: [{ name: 'desktop-chrome', use: { ...devices['Desktop Chrome'] } }],
})
