import { expect, test } from '@playwright/test'

/**
 * Post-deploy smoke vs the LIVE player (#43). Validates the deployed closelistening.app:
 * the coming-soon gate holds for the public, preview users reach the real app, key routes
 * render, the Google sign-in entrypoint is wired end-to-end, and the backend is healthy.
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
    await expect(page.getByText('Learning Player')).toHaveCount(0)
  } finally {
    await ctx.close()
  }
})

test.describe('preview surface', () => {
  test.skip(!gated, 'set PLAYER_PREVIEW_PASS to run the gated live specs')

  test('preview users reach the real app home', async ({ page }) => {
    // /preview issues the basic-auth challenge (satisfied by httpCredentials), sets the
    // preview cookie, and 302s to /.
    await page.goto('/preview')
    await expect(page).toHaveURL(/closelistening\.app\/?$/)
    await expect(page.getByText('Learning Player').first()).toBeVisible()
    await expect(page.getByText("Find any moment you've heard.")).toBeVisible()
  })

  test('catalog route renders', async ({ page }) => {
    await page.goto('/preview')
    await page.goto('/catalog')
    await expect(page).toHaveURL(/\/catalog$/)
    await expect(page.getByRole('link', { name: 'Learning Player' })).toBeVisible()
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
