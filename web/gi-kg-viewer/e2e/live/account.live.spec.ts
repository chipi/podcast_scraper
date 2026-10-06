import { createHmac } from 'node:crypto'
import { expect, test } from '@playwright/test'

/**
 * Post-deploy live smoke for the AUTHENTICATED operator surface at operator.closelistening.app.
 *
 * The operator viewer is a login wall behind the coming-soon gate (real Google OAuth on prod,
 * ≥creator role required — a headless smoke can't complete Google). So we authenticate as
 * dedicated **prod test accounts** by minting the same HMAC-signed session token the app issues
 * (`app_sessions.sign`) — identical mechanism to the player's `account.live.spec.ts`, and the
 * operator deploy uses the SAME `APP_SESSION_SECRET` (`deploy-operator.yml` sets it from
 * `PLAYER_APP_SESSION_SECRET`).
 *
 * The public operator plane (`_OPERATOR_PUBLIC_READ_ROUTES`) is READ-ONLY, so — unlike the player —
 * there is no reversible-write round-trip. The value here is: authenticated reads succeed for a
 * creator, and the **role boundary** holds (a creator is 403'd from admin-only routes).
 *
 * CREATOR-ONLY by design: we deliberately do NOT mint an admin session in CI. An admin token from a
 * CI secret that can read the admin/user-management plane is unnecessary overhead + a security gap for
 * a smoke; a creator token proves the operator surface works AND that the admin boundary holds.
 *
 * Requires (skips cleanly otherwise):
 *   PLAYER_APP_SESSION_SECRET        the prod session-signing secret (already a deploy secret)
 *   OPERATOR_SMOKE_CREATOR_USER_ID   stored id (u_…) of a seeded CREATOR test user
 */
const secret = process.env.PLAYER_APP_SESSION_SECRET || ''
const creatorId = process.env.OPERATOR_SMOKE_CREATOR_USER_ID || ''
// The coming-soon gate's PRIMARY mechanism is the cl_op_preview COOKIE (infra/caddy/operator.caddy
// `@preview_ok`), sent on XHR/API alike; basic-auth is only a curl fallback. A Bearer call must
// therefore clear the gate via the cookie — an explicit `Authorization: Bearer` would override the
// Basic that httpCredentials sends. So each request test primes /preview first (httpCredentials
// satisfies the basic-auth challenge, the gate Set-Cookies cl_op_preview into the shared jar), then
// the Bearer calls pass the gate by cookie. Needs the gate password present, hence it gates the skip.
const gatePass = process.env.OPERATOR_PREVIEW_PASS || process.env.PLAYER_PREVIEW_PASS || ''

/** Mint the app's session token — byte-matches app_sessions.sign (urlsafe-b64, HMAC-SHA256,
 *  compact + sorted-keys JSON). Integer `iat` avoids any JS/Python float-repr mismatch. */
function mint(userId: string): string {
  const b64url = (b: Buffer) => b.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  const json = JSON.stringify({ iat: Math.floor(Date.now() / 1000), user_id: userId })
  const body = b64url(Buffer.from(json, 'utf-8'))
  return `${body}.${b64url(createHmac('sha256', secret).update(body).digest())}`
}
const authFor = (id: string) => ({ Authorization: `Bearer ${mint(id)}` })

// A cheap, always-present slice of the ≥creator operator read plane (avoid heavy binary/media).
const CREATOR_READ_ROUTES = [
  '/api/app/me',
  '/api/corpus/digest',
  '/api/corpus/coverage',
  '/api/corpus/stats',
  '/api/index/stats',
  '/api/corpus/trending',
  '/api/corpus/persons/top',
]

test.describe('operator — authed as creator', () => {
  test.skip(
    !(secret && creatorId && gatePass),
    'set PLAYER_APP_SESSION_SECRET + OPERATOR_SMOKE_CREATOR_USER_ID + gate password to run',
  )
  // Seed the cl_op_preview gate cookie into the request jar so the Bearer calls below clear the gate.
  test.beforeEach(async ({ request }) => {
    await request.get('/preview')
  })

  test('the minted creator session authenticates (≥creator)', async ({ request }) => {
    const me = await request.get('/api/app/me', { headers: authFor(creatorId) })
    expect(me.status()).toBe(200)
    const role = (await me.json()).role
    expect(['creator', 'admin'], `role was ${role}`).toContain(role)
  })

  test('the ≥creator operator read plane returns 200', async ({ request }) => {
    for (const path of CREATOR_READ_ROUTES) {
      const resp = await request.get(path, { headers: authFor(creatorId) })
      expect(resp.status(), `${path} should be 200 for a creator`).toBe(200)
    }
  })

  test('a creator is 403 from admin-only routes (role boundary)', async ({ request }) => {
    // We assert the boundary with the CREATOR token (must be denied) — no admin token is minted.
    const resp = await request.get('/api/app/admin/users', { headers: authFor(creatorId) })
    expect(resp.status(), 'creator must NOT reach admin/users').toBe(403)
  })

  test('a creator session boots the viewer, not the login wall', async ({ browser, baseURL }) => {
    const origin = baseURL || 'https://operator.closelistening.app'
    // A fresh context does NOT inherit the config's `use.httpCredentials`; pass the gate creds so
    // /preview's basic-auth challenge is satisfied and the gate sets cl_op_preview.
    const ctx = await browser.newContext({
      serviceWorkers: 'block',
      httpCredentials: {
        username: process.env.OPERATOR_PREVIEW_USER || 'marko',
        password: gatePass,
        origin,
      },
    })
    const host = new URL(origin).hostname
    await ctx.addCookies([
      { name: 'lp_session', value: mint(creatorId), domain: host, path: '/', httpOnly: true, secure: true },
    ])
    try {
      const page = await ctx.newPage()
      // /preview sets the gate cookie, then the SPA boots with a ≥creator session → viewer, not login.
      await page.goto('/preview')
      await expect(page.getByText('Sign in to explore the knowledge graph')).toHaveCount(0)
      await expect(page.getByTestId('login-button')).toHaveCount(0)
    } finally {
      await ctx.close()
    }
  })
})
