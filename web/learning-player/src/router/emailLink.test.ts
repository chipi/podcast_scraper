/**
 * `email_link_opened` — which links people click in the emails we send (operator 2026-10-05).
 * Driven through the real router and gate: the event is emitted there because only the gate knows
 * both where the link points and whether the person is signed in.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

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
      throw new Error('transport')
    }),
  }
})
vi.mock('../services/analytics', async (orig) => {
  const actual = await orig<typeof import('../services/analytics')>()
  return { ...actual, track: vi.fn() }
})

import { isNative } from '../services/native'
import { setAuthToken } from '../services/api'
import { track } from '../services/analytics'
import { router } from './index'

const asMock = (fn: unknown) => fn as unknown as ReturnType<typeof vi.fn>
const clicks = () => asMock(track).mock.calls.filter((c) => c[0] === 'email_link_opened').map((c) => c[1])

const FROM_DIGEST =
  '/episode/ep-1?t=65&utm_source=email&utm_campaign=your_week_digest&utm_content=episode'

beforeEach(async () => {
  sessionStorage.clear()
  setActivePinia(createPinia())
  asMock(isNative).mockReturnValue(false)
  setAuthToken(null)
  await router.replace('/welcome')
  asMock(track).mockClear()
})

describe('email_link_opened', () => {
  it('a signed-out click is counted at the gate, then sent to sign in with the link intact', async () => {
    await router.push(FROM_DIGEST)
    expect(clicks()).toEqual([
      { campaign: 'your_week_digest', element: 'episode', target: 'player', signed_in: false },
    ])
    expect(router.currentRoute.value.name).toBe('landing')
    expect(router.currentRoute.value.query.redirect).toBe(FROM_DIGEST)
  })

  it('a signed-in click is counted as signed in, and the page opens', async () => {
    asMock(isNative).mockReturnValue(true) // a stored token admits, per the gate's native rule
    setAuthToken('signed-token')
    await router.push(FROM_DIGEST)
    expect(clicks()).toEqual([
      { campaign: 'your_week_digest', element: 'episode', target: 'player', signed_in: true },
    ])
    expect(router.currentRoute.value.name).toBe('player')
  })

  it('a reload of the same link is not a second click', async () => {
    await router.push(FROM_DIGEST)
    await router.replace('/welcome')
    await router.push(FROM_DIGEST)
    expect(clicks()).toHaveLength(1)
  })

  it('an unknown value is reported as other — a hand-edited link cannot inject text', async () => {
    await router.push('/topic/topic%3Aai?utm_source=email&utm_campaign=hello_world&utm_content=x')
    expect(clicks()).toEqual([{ campaign: 'other', element: 'other', target: 'topic', signed_in: false }])
  })

  it('a visit that did not come from an email reports nothing', async () => {
    await router.push('/episode/ep-1?t=65')
    await router.push('/episode/ep-2?utm_source=newsletter')
    expect(clicks()).toEqual([])
  })
})
