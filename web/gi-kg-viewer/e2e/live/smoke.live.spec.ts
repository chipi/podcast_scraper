import { expect, test } from '@playwright/test'

/**
 * Post-deploy smoke vs the LIVE operator viewer (#43). Validates the deployed
 * operator.closelistening.app: the coming-soon gate holds for the public, preview users reach
 * the viewer (which is a login wall), the Google sign-in entrypoint is wired end-to-end with
 * the correct HTTPS callback, and the backend is healthy.
 *
 * Runs under playwright.live.config.ts (baseURL = the live origin; preview basic-auth via
 * httpCredentials). The gated specs skip when no preview password is set.
 *
 * SCOPE LIMIT (see the config header): the operator viewer renders nothing until a ≥creator
 * Google session exists, and prod runs the real provider (the mock ?as= login can never ship —
 * app_oauth.py). So we assert the sign-in REDIRECT, not a completed login. The authed viewer +
 * corpus flow is covered by stack-test under the mock provider.
 */

const gated = Boolean(process.env.OPERATOR_PREVIEW_PASS || process.env.PLAYER_PREVIEW_PASS)

test('coming-soon gate holds for the public (no preview creds)', async ({ browser }) => {
  // A fresh context WITHOUT credentials must see the marketing gate, never the viewer.
  const ctx = await browser.newContext({ httpCredentials: undefined, serviceWorkers: 'block' })
  try {
    const page = await ctx.newPage()
    const resp = await page.goto('/')
    expect(resp?.status()).toBe(200)
    await expect(page.getByText('Coming soon')).toBeVisible()
    // The viewer's sign-in button must NOT be reachable without the gate.
    await expect(page.getByTestId('login-button')).toHaveCount(0)
  } finally {
    await ctx.close()
  }
})

test.describe('preview surface', () => {
  test.skip(!gated, 'set OPERATOR_PREVIEW_PASS (or PLAYER_PREVIEW_PASS) to run the gated live specs')

  test('preview users reach the viewer login wall', async ({ page }) => {
    // Past the doorman (basic-auth) the SPA boots, sees no ≥creator session, and renders the
    // LoginView — the real-provider "Sign in" button, not the mock picker.
    await page.goto('/')
    await expect(page.getByText('Sign in to explore the knowledge graph')).toBeVisible()
    await expect(page.getByTestId('login-button')).toBeVisible()
  })

  test('sign-in entrypoint 307s to Google OAuth with the HTTPS operator callback', async ({
    page,
  }) => {
    await page.goto('/')
    // Clicking sign-in calls auth.login() -> location.assign('/api/app/auth/login?grant=creator'),
    // which the backend 307s to Google's consent screen. Assert that redirect directly — the
    // exact chain the redirect_uri_mismatch broke — rather than loading Google's heavy page.
    const respPromise = page.waitForResponse((r) => r.url().includes('/api/app/auth/login'), {
      timeout: 25_000,
    })
    // dispatchEvent fires the native click WITHOUT Playwright waiting for the cross-origin
    // Google navigation to settle.
    await page.getByTestId('login-button').dispatchEvent('click')
    const resp = await respPromise
    expect(resp.status()).toBe(307)
    const location = (await resp.headerValue('location')) ?? ''
    expect(location).toContain('accounts.google.com')
    // Regression guard for 2026-07-25: the callback must be HTTPS on the operator host. The
    // whole encoded scheme+host+path is asserted, so an http:// redirect (the bug) fails here.
    expect(location).toContain(
      'redirect_uri=https%3A%2F%2Foperator.closelistening.app%2Fapi%2Fapp%2Fauth%2Fcallback',
    )
  })

  test('backend health is green', async ({ request }) => {
    const resp = await request.get('/api/health')
    expect(resp.status()).toBe(200)
  })
})
