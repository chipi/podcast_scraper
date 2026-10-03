/**
 * `screen_view` fires once per SCREEN, not once per URL change (prod 2026-10-04: seven for one
 * episode whose path never changed). Driven through the real router so the hook, not a regex over
 * its source, is what is tested.
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

import { track } from '../services/analytics'
import { router } from './index'

const screenViews = () =>
  (track as unknown as ReturnType<typeof vi.fn>).mock.calls.filter((c) => c[0] === 'screen_view')

describe('screen_view', () => {
  beforeEach(async () => {
    setActivePinia(createPinia())
    await router.replace('/welcome')
    ;(track as unknown as ReturnType<typeof vi.fn>).mockClear()
  })

  it('is not re-sent when only the query or hash changes', async () => {
    await router.push('/login')
    await router.replace('/login?next=%2F')
    await router.replace('/login?next=%2F#x')
    expect(screenViews()).toEqual([['screen_view', { screen: 'login' }]])
  })

  it('is sent again when the path changes', async () => {
    await router.push('/login')
    await router.push('/welcome')
    expect(screenViews().map((c) => c[1])).toEqual([{ screen: 'login' }, { screen: 'landing' }])
  })
})
