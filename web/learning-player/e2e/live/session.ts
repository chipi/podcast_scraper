import { createHmac } from 'node:crypto'
import type { Browser, BrowserContext } from '@playwright/test'

/**
 * Minting a prod session for the live smokes.
 *
 * A headless smoke cannot complete a real Google sign-in, so it authenticates as a dedicated **prod
 * test account** by minting the same HMAC-signed token the app issues (`app_sessions.sign`) from the
 * session secret plus the test user's id.
 *
 * ## Why this is shared rather than copied
 *
 * It lived inside `account.live.spec.ts`. RFC-120 (#1940) then put `Depends(get_current_user)` on
 * `/api/app/episodes` and its siblings, and `privacy-floor.live.spec.ts` — which primes only the
 * coming-soon gate and never signs in — started getting 401 where it asserts 200. It failed the
 * post-deploy smoke on two consecutive prod deploys.
 *
 * The second spec needs the identical token, and "identical" is the operative word: the encoding has
 * to byte-match the Python side. Two copies of that is two things to keep in step with
 * `app_sessions.sign`, and the copy nobody is currently debugging is the one that drifts.
 */

/** The prod session-signing secret — already a deploy secret. */
export const sessionSecret = process.env.PLAYER_APP_SESSION_SECRET || ''

/** The id of a test user SEEDED in the prod user store. */
export const smokeUserId = process.env.PLAYER_SMOKE_USER_ID || ''

/**
 * The coming-soon gate password (`infra/caddy/player.caddy`).
 *
 * The gate's primary mechanism is the `cl_preview` COOKIE. An explicit `Authorization: Bearer`
 * overrides the Basic that `httpCredentials` sends, so a Bearer call cannot lean on basic-auth —
 * every such spec must prime `/preview` first so the gate sets that cookie into the shared jar.
 */
export const gatePass = process.env.PLAYER_PREVIEW_PASS || ''

/** True when a live spec has everything it needs to authenticate; otherwise it should skip. */
export const canMintSession = Boolean(sessionSecret && smokeUserId && gatePass)

function b64url(b: Buffer): string {
  return b.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

/**
 * Mint the app's session token.
 *
 * Must byte-match `app_sessions.sign`: urlsafe-b64, HMAC-SHA256, compact JSON with SORTED keys
 * (`iat` before `user_id`). An INTEGER `iat` avoids any JS/Python float-repr mismatch.
 */
export function mintSession(): string {
  const json = JSON.stringify({ iat: Math.floor(Date.now() / 1000), user_id: smokeUserId })
  const body = b64url(Buffer.from(json, 'utf-8'))
  const sig = b64url(createHmac('sha256', sessionSecret).update(body).digest())
  return `${body}.${sig}`
}

/** `Authorization` header carrying a freshly minted session. */
export function bearer(): { Authorization: string } {
  return { Authorization: `Bearer ${mintSession()}` }
}

/** The preview gate's basic-auth user (`infra/caddy/player.caddy`). */
export const gateUser = process.env.PLAYER_PREVIEW_USER || 'marko'

/**
 * A browser context that is through the gate AND signed in.
 *
 * Two DIFFERENT doors, and every live UI spec has to open both — which is the thing RFC-120 (#1940)
 * changed and the live specs did not follow:
 *
 *   1. the coming-soon gate (`cl_preview`, basic-auth on `/preview`) — infra, fronts everything
 *   2. the app session (`lp_session`) — the app itself, now DENY-BY-DEFAULT
 *
 * Before login-first, opening door 1 was enough: `/` rendered the app. Now a visitor who clears the
 * gate but has no session is redirected to `/welcome`, so every spec that navigated to app content
 * after only `/preview` began asserting against the lure landing. That is what failed the
 * post-deploy smoke on two consecutive prod deploys — deterministically, which is why the retries
 * did not rescue it.
 *
 * A fresh context does NOT inherit the config's `use.httpCredentials`, so the gate creds are passed
 * explicitly here.
 */
export async function signedInContext(browser: Browser, origin: string): Promise<BrowserContext> {
  const ctx = await browser.newContext({
    serviceWorkers: 'block',
    httpCredentials: { username: gateUser, password: gatePass, origin },
  })
  await addSessionCookie(ctx, origin)
  return ctx
}

/**
 * Sign an EXISTING context in — for specs that use the shared `page` fixture and only need door 2.
 *
 * The fixture's context already carries the config's `httpCredentials`, so it can clear the gate on
 * its own; all it lacks is the session. Adding the cookie in a `beforeEach` keeps those specs as
 * they are instead of rebuilding every test around a fresh context.
 */
export async function addSessionCookie(ctx: BrowserContext, origin: string): Promise<void> {
  await ctx.addCookies([
    {
      name: 'lp_session',
      value: mintSession(),
      domain: new URL(origin).hostname,
      path: '/',
      httpOnly: true,
      secure: true,
    },
  ])
}
