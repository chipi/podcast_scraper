import { createHmac } from 'node:crypto'
import { expect, test } from '@playwright/test'

/**
 * Post-deploy live smoke for the PER-USER surfaces (Collections / Library / Queue) against the
 * deployed player. A headless smoke can't complete a real Google sign-in, so it authenticates as a
 * dedicated **prod test account** by minting the same HMAC-signed session token the app issues
 * (`app_sessions.sign`) from the session secret + the test user's id.
 *
 * Requires (skips cleanly otherwise):
 *   PLAYER_APP_SESSION_SECRET  the prod session-signing secret (already a deploy secret)
 *   PLAYER_SMOKE_USER_ID       the id of a test user SEEDED in the prod user store
 *
 * All writes are REVERSIBLE (create→assert→delete) and scoped to the test account, so the smoke
 * never leaves residue on a real user.
 */
const secret = process.env.PLAYER_APP_SESSION_SECRET || ''
const userId = process.env.PLAYER_SMOKE_USER_ID || ''
// The coming-soon gate's PRIMARY mechanism is the cl_preview COOKIE (infra/caddy/player.caddy),
// attached to every same-origin request. A Bearer call clears the gate via the cookie — an explicit
// `Authorization: Bearer` overrides the Basic that httpCredentials sends, so we can't lean on
// basic-auth here. Each test primes /preview first (httpCredentials satisfies the basic-auth
// challenge; the gate Set-Cookies cl_preview into the shared jar), then Bearer calls pass by cookie.
// Needs the gate password present, hence it gates the skip.
const gatePass = process.env.PLAYER_PREVIEW_PASS || ''
const enabled = Boolean(secret && userId && gatePass)

/** Mint the app's session token — must byte-match app_sessions.sign (urlsafe-b64, HMAC-SHA256,
 *  compact + sorted-keys JSON). An INTEGER `iat` avoids any JS/Python float-repr mismatch. */
function mintSession(): string {
  const b64url = (b: Buffer) => b.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  // Keys MUST be alphabetically sorted to match json.dumps(sort_keys=True): iat < user_id.
  const json = JSON.stringify({ iat: Math.floor(Date.now() / 1000), user_id: userId })
  const body = b64url(Buffer.from(json, 'utf-8'))
  const sig = b64url(createHmac('sha256', secret).update(body).digest())
  return `${body}.${sig}`
}

const bearer = () => ({ Authorization: `Bearer ${mintSession()}` })

test.describe('per-user surfaces (test account)', () => {
  test.skip(
    !enabled,
    'set PLAYER_APP_SESSION_SECRET + PLAYER_SMOKE_USER_ID (seeded test user) + gate password to run',
  )
  // Seed the cl_preview gate cookie into the request jar so the Bearer calls below clear the gate.
  test.beforeEach(async ({ request }) => {
    await request.get('/preview')
  })

  test('the minted session authenticates (not 401)', async ({ request }) => {
    const me = await request.get('/api/app/me', { headers: bearer() })
    expect(me.status(), 'minted session should resolve to the test user').toBe(200)
    expect((await me.json()).user_id).toBe(userId)
  })

  test('Collections round-trips (create → list → delete)', async ({ request }) => {
    const name = `smoke-${Date.now()}`
    const created = await request.post('/api/app/collections', { headers: bearer(), data: { name } })
    expect(created.status(), 'POST /collections is 201 Created').toBe(201)
    const id = (await created.json()).id
    try {
      const list = await request.get('/api/app/collections', { headers: bearer() })
      expect(list.status()).toBe(200)
      const items = ((await list.json()).items ?? []) as Array<{ id: string }>
      expect(items.some((c) => c.id === id)).toBe(true)
    } finally {
      // Always clean up, even if an assertion above failed — never leave residue.
      const del = await request.delete(`/api/app/collections/${id}`, { headers: bearer() })
      expect(del.status()).toBe(200)
      const after = ((await del.json()).items ?? []) as Array<{ id: string }>
      expect(after.some((c) => c.id === id)).toBe(false)
    }
  })

  test('signed-in Library renders its tabs', async ({ browser, baseURL }) => {
    const origin = baseURL || 'https://closelistening.app'
    // A fresh context does NOT inherit the config's `use.httpCredentials`, so pass the gate creds
    // explicitly — /preview's basic-auth challenge must be satisfied to obtain cl_preview.
    const ctx = await browser.newContext({
      serviceWorkers: 'block',
      httpCredentials: { username: process.env.PLAYER_PREVIEW_USER || 'marko', password: gatePass, origin },
    })
    // Web auth is the cookie; set the same minted token as the lp_session cookie.
    const host = new URL(origin).hostname
    await ctx.addCookies([
      { name: 'lp_session', value: mintSession(), domain: host, path: '/', httpOnly: true, secure: true },
    ])
    try {
      const page = await ctx.newPage()
      // /preview clears the coming-soon gate (sets cl_preview) before the app can render.
      await page.goto('/preview')
      await page.goto('/library')
      // A signed-in Library shows its tabs (Saved · Following · Collections · Revisit).
      await expect(page.getByRole('button', { name: 'Collections' })).toBeVisible()
      await expect(page.getByRole('button', { name: 'Saved' })).toBeVisible()
    } finally {
      await ctx.close()
    }
  })
})
