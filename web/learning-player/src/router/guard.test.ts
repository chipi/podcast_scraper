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
    getAuthToken: vi.fn(() => null),
    getMe: vi.fn(async () => {
      throw new Error('transport') // NOT a 401 — inconclusive, so auth keeps user null but no sign-out
    }),
  }
})

import { isNative } from '../services/native'
import { getAuthToken } from '../services/api'
import { router } from './index'

const asMock = (fn: unknown) => fn as unknown as ReturnType<typeof vi.fn>

beforeEach(async () => {
  setActivePinia(createPinia())
  asMock(isNative).mockReturnValue(false)
  asMock(getAuthToken).mockReturnValue(null)
  await router.replace('/welcome')
})

describe('login-first guard — native token hardening', () => {
  it('web, no session: a protected route redirects to the landing', async () => {
    await router.push('/library')
    expect(router.currentRoute.value.name).toBe('landing')
  })

  it('native WITH a stored token but an unconfirmed session: reaches the protected route (no strand)', async () => {
    asMock(isNative).mockReturnValue(true)
    asMock(getAuthToken).mockReturnValue('signed-token')
    await router.push('/library')
    expect(router.currentRoute.value.name).toBe('library')
  })

  it('native WITHOUT a token: still redirects to the landing', async () => {
    asMock(isNative).mockReturnValue(true)
    asMock(getAuthToken).mockReturnValue(null)
    await router.push('/library')
    expect(router.currentRoute.value.name).toBe('landing')
  })
})
