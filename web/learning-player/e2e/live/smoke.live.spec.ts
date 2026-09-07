import { expect, test } from '@playwright/test'
import { canMintSession, signedInContext } from './session'

/**
 * Post-deploy smoke vs the LIVE player (#43). Validates the deployed closelistening.app:
 * the coming-soon gate holds for the public, a preview visitor meets the lure landing, a SIGNED-IN
 * preview user reaches the real app, key routes render, the Google sign-in entrypoint is wired
 * end-to-end, and the backend is healthy.
 *
 * ## Two doors, since RFC-120 (#1940)
 *
 * The coming-soon gate (infra) and the app session (deny-by-default) are separate. Clearing the
 * gate used to be enough for `/` to render the app; now it yields `/welcome`. These specs cleared
 * only the gate and then asserted on app content, so they failed the post-deploy smoke on two
 * consecutive prod deploys — deterministically, which is why the retries did not rescue them.
 *
 * Runs under playwright.live.config.ts (baseURL = the live origin; preview basic-auth via
 * httpCredentials). The gated specs skip when PLAYER_PREVIEW_PASS is unset.
 */

const gated = Boolean(process.env.PLAYER_PREVIEW_PASS)

test('coming-soon gate holds for the public (no preview creds)', async ({ browser }) => {
  // A fresh context WITHOUT credentials must see the marketing gate, never the app.
  const ctx = await browser.newContext({ httpCredentials: undefined, serviceWorkers: 'block' })
  try {
    const page = await ctx.newPage()
    const resp = await page.goto('/')
    expect(resp?.status()).toBe(200)
    await expect(page.getByText('Coming soon')).toBeVisible()
    // App-only marker (the home hero) must be absent — the "Close Listening" brand can also appear
    // on the marketing gate, so assert on something that ONLY the real app renders.
    await expect(page.getByText("Find any moment you've heard.")).toHaveCount(0)
  } finally {
    await ctx.close()
  }
})

test.describe('preview surface', () => {
  test.skip(!gated, 'set PLAYER_PREVIEW_PASS to run the gated live specs')

  test('a preview visitor who is NOT signed in meets the lure landing', async ({ page }) => {
    // /preview issues the basic-auth challenge (satisfied by httpCredentials), sets the preview
    // cookie, and 302s to /. Under login-first the router then sends a session-less visitor to
    // /welcome. This test used to assert the Home hero here; that is the pre-RFC-120 behaviour and
    // asserting it is what turned the prod smoke red.
    await page.goto('/preview')
    await expect(page).toHaveURL(/\/welcome/)
    await expect(page.getByTestId('landing-cta-primary')).toBeVisible()
    // The app's own hero must NOT be here — that is the marker separating landing from app.
    await expect(page.getByText("Find any moment you've heard.")).toHaveCount(0)
  })

  test('a SIGNED-IN preview user reaches the real app home', async ({ browser, baseURL }) => {
    test.skip(!canMintSession, 'needs PLAYER_APP_SESSION_SECRET + PLAYER_SMOKE_USER_ID')
    const ctx = await signedInContext(browser, baseURL || 'https://closelistening.app')
    try {
      const page = await ctx.newPage()
      await page.goto('/preview')
      await page.goto('/')
      await expect(page.getByText("Find any moment you've heard.")).toBeVisible()
    } finally {
      await ctx.close()
    }
  })

  test('catalog route renders for a signed-in user', async ({ browser, baseURL }) => {
    test.skip(!canMintSession, 'needs PLAYER_APP_SESSION_SECRET + PLAYER_SMOKE_USER_ID')
    const ctx = await signedInContext(browser, baseURL || 'https://closelistening.app')
    try {
      const page = await ctx.newPage()
      await page.goto('/preview')
      await page.goto('/catalog')
      await expect(page).toHaveURL(/\/catalog$/)
      await expect(page.getByRole('link', { name: 'Close Listening' })).toBeVisible()
    } finally {
      await ctx.close()
    }
  })

  test('sign-in entrypoint 307s to Google OAuth', async ({ page }) => {
    await page.goto('/preview')
    await page.goto('/login')
    await expect(page.getByRole('heading', { name: /Sign in to your library/ })).toBeVisible()
    // Clicking the prod sign-in button navigates top-level to /api/app/auth/login, which the
    // backend 307s to Google's consent screen. Assert that redirect directly — the exact chain
    // the launch bugs broke — rather than fully loading Google's heavy (flaky) consent page.
    const respPromise = page.waitForResponse((r) => r.url().includes('/api/app/auth/login'), {
      timeout: 25_000,
    })
    // dispatchEvent fires the native click (Vue's @click -> auth.login() -> location.assign)
    // WITHOUT Playwright waiting for the ensuing cross-origin Google navigation to settle.
    await page.getByRole('button', { name: 'Sign in' }).dispatchEvent('click')
    const resp = await respPromise
    expect(resp.status()).toBe(307)
    const location = (await resp.headerValue('location')) ?? ''
    expect(location).toContain('accounts.google.com')
    expect(location).toContain('closelistening.app%2Fapi%2Fapp%2Fauth%2Fcallback')
  })

  test('backend health is green', async ({ request }) => {
    const resp = await request.get('/api/health')
    expect(resp.status()).toBe(200)
  })
})
