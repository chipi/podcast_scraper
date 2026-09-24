/**
 * Login-first guard (RFC-120) regression tests, focused on the 2026-09-15 native desync:
 * a returning user whose cold-start `GET /me` TRANSPORT-fails (not a 401) was stranded on the public
 * landing even though the header showed them logged in. The guard now treats a stored native bearer
 * token as signed-in for routing, so that user reaches the app; reads are open, and a genuinely dead
 * token surfaces as a 401 on the first authed call.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

// Simulate the desync's precondition: no device snapshot + a `getMe` that transport-fails, so the
// auth store ends up unauthenticated (user === null) after ensureLoaded().
vi.mock('../services/deviceStore', () => ({
  getDeviceJson: vi.fn(async () => null),
  setDeviceJson: vi.fn(async () => {}),
  removeDeviceKey: vi.fn(async () => {}),
}))
vi.mock('../services/native', () => ({ isNative: vi.fn(() => false) }))
vi.mock('../services/api', async (orig) => {
  const actual = await orig<typeof import('../services/api')>()
  return {
    ...actual,
    getMe: vi.fn(async () => {
      throw new Error('transport') // NOT a 401 — inconclusive, so auth keeps user null but no sign-out
    }),
  }
})

import { isNative } from '../services/native'
import { setAuthToken } from '../services/api'
import { router } from './index'

const asMock = (fn: unknown) => fn as unknown as ReturnType<typeof vi.fn>

// The token is driven through the REAL `setAuthToken` rather than by stubbing `getAuthToken`.
// The guard now asks `auth.hasSession`, a Pinia getter — i.e. a Vue computed — which only
// re-evaluates when a REACTIVE dependency changes. A stubbed accessor is not one, so the getter
// would keep returning whatever it computed during this hook's `router.replace` and the token set
// inside a test would never be observed. Driving the real setter exercises the production path and
// invalidates the computed the same way a login does.
beforeEach(async () => {
  setActivePinia(createPinia())
  asMock(isNative).mockReturnValue(false)
  setAuthToken(null)
  await router.replace('/welcome')
})

describe('login-first guard — native token hardening', () => {
  it('web, no session: a protected route redirects to the landing', async () => {
    await router.push('/library')
    expect(router.currentRoute.value.name).toBe('landing')
  })

  it('native WITH a stored token but an unconfirmed session: reaches the protected route (no strand)', async () => {
    asMock(isNative).mockReturnValue(true)
    setAuthToken('signed-token')
    await router.push('/library')
    expect(router.currentRoute.value.name).toBe('library')
  })

  it('native WITHOUT a token: still redirects to the landing', async () => {
    asMock(isNative).mockReturnValue(true)
    setAuthToken(null)
    await router.push('/library')
    expect(router.currentRoute.value.name).toBe('landing')
  })
})


describe('a token without an identity must never lock you out (2026-09-24)', () => {
  /*
   * The lockout, found on device. `hasSession` is true on a stored token ALONE, so a token the
   * server no longer honours produced a session that could not name its own user: every authed
   * call 401'd, the app showed nothing, and `/login` — the one screen that could fix it —
   * redirected to Home. The dev picker "never rendered" across a dozen runs for this reason.
   *
   * The rule: a token may ADMIT you to authed routes, but only a RESOLVED identity may block you
   * from signing in.
   */
  it('lets a token-only session reach /login instead of bouncing it to Home', async () => {
    asMock(isNative).mockReturnValue(true)
    setAuthToken('stale-token-the-server-no-longer-honours')

    await router.replace('/login')

    expect(router.currentRoute.value.name).toBe('login')
  })

  it('still bounces a RESOLVED identity away from /login', async () => {
    // The original behaviour, which is correct and must survive: someone genuinely signed in has
    // no business on the login page.
    asMock(isNative).mockReturnValue(true)
    setAuthToken('any')
    const { useAuthStore } = await import('../stores/auth')
    useAuthStore().user = { user_id: 'u_1', email: 'a@b.c', name: 'A' } as never

    await router.replace('/login')

    expect(router.currentRoute.value.name).toBe('home')
  })

  it('a token-only session is still ADMITTED to authed routes', async () => {
    // The 2026-09-15 desync fix must not regress: a transport failure cannot strand a returning
    // user on the landing.
    asMock(isNative).mockReturnValue(true)
    setAuthToken('token')

    await router.replace('/library')

    expect(router.currentRoute.value.name).toBe('library')
  })
})

