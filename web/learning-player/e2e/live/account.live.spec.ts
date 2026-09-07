import { expect, test } from '@playwright/test'
import { bearer, canMintSession, gatePass, mintSession, smokeUserId } from './session'

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
// Session minting moved to ./session.ts — `privacy-floor.live.spec.ts` needs the identical token,
// and the encoding has to byte-match the Python side, so one copy rather than two.
const userId = smokeUserId
const enabled = canMintSession

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
      // Library's tabs are `role="tab"` since #1594 item 7 — they previously carried NO role at
      // all, which is why `getByRole('button')` matched them.
      await expect(page.getByRole('tab', { name: 'Collections' })).toBeVisible()
      await expect(page.getByRole('tab', { name: 'Saved' })).toBeVisible()
    } finally {
      await ctx.close()
    }
  })
})
