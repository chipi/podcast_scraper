import { expect, test } from '@playwright/test'

/**
 * Post-deploy smoke vs the LIVE operator host: it is CLOSED (operator decision 2026-10-05 — the
 * operator surface is tailnet-only; infra/caddy/operator.caddy proxies nothing to the app).
 *
 * What a pass proves: DNS + Cloudflare + TLS + the edge serve operator.closelistening.app, and NO
 * path reaches the operator app — not the SPA, not the sign-in, not the backend, not the admin
 * routes — with or without the old /preview basic-auth or preview cookie. A reopened doorman, a
 * stray reverse_proxy, or a vhost that fell back to another site fails here.
 *
 * Needs no secrets: the closed site ignores credentials, so the spec sends a made-up Basic header
 * and a made-up preview cookie to prove they open nothing.
 */

const CLOSED_PATHS = [
  '/',
  '/preview',
  '/index.html',
  '/api/health',
  '/api/app/auth/login',
  '/api/app/me',
  '/api/app/admin/users',
  '/api/feeds',
  '/api/feeds/overrides',
]

const OLD_KEYS = {
  Authorization: `Basic ${Buffer.from('marko:not-a-real-password').toString('base64')}`,
  Cookie: 'cl_op_preview=not-a-real-cookie',
}

for (const path of CLOSED_PATHS) {
  test(`${path} answers the coming-soon page, with or without the old gate keys`, async ({
    playwright,
    baseURL,
  }) => {
    for (const headers of [{}, OLD_KEYS]) {
      const ctx = await playwright.request.newContext({ baseURL, extraHTTPHeaders: headers })
      try {
        const resp = await ctx.get(path, { maxRedirects: 0 })
        // 200 + the page: never a 401 challenge (the doorman), a 302 (its redirect), a 307 (the
        // OAuth hand-off), or JSON from the backend.
        expect(resp.status(), `${path} ${JSON.stringify(Object.keys(headers))}`).toBe(200)
        expect(resp.headers()['content-type'] ?? '').toContain('text/html')
        const body = await resp.text()
        expect(body).toContain('Coming soon')
        expect(body).not.toContain('<div id="app">')
      } finally {
        await ctx.dispose()
      }
    }
  })
}

test('the browser sees coming-soon and no sign-in', async ({ browser }) => {
  const ctx = await browser.newContext({ serviceWorkers: 'block' })
  try {
    const page = await ctx.newPage()
    const resp = await page.goto('/')
    expect(resp?.status()).toBe(200)
    await expect(page.getByText('Coming soon')).toBeVisible()
    await expect(page.getByTestId('login-button')).toHaveCount(0)
  } finally {
    await ctx.close()
  }
})
