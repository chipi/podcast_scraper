import { beforeEach, describe, expect, it, vi } from 'vitest'

// A tapped push opens what it points at: the episode, or What's new for several (2026-10-09).

let native = true
type Action = { actionId: string; notification: { data?: Record<string, unknown> } }
let onAction: ((a: Action) => void) | null = null

vi.mock('@capacitor/core', () => ({
  Capacitor: { isNativePlatform: () => native },
}))
vi.mock('@capacitor/push-notifications', () => ({
  PushNotifications: {
    addListener: vi.fn(async (_name: string, cb: (a: Action) => void) => {
      onAction = cb
      return { remove: async () => {} }
    }),
  },
}))

const { initPushTaps, pushTapPath } = await import('./pushTaps')
const { PushNotifications } = await import('@capacitor/push-notifications')

beforeEach(() => {
  native = true
  onAction = null
  vi.mocked(PushNotifications.addListener).mockClear()
})

describe('initPushTaps', () => {
  it('opens the episode a single-episode push points at', async () => {
    const navigate = vi.fn()
    await initPushTaps(navigate)
    expect(PushNotifications.addListener).toHaveBeenCalledWith(
      'pushNotificationActionPerformed',
      expect.any(Function),
    )
    onAction!({ actionId: 'tap', notification: { data: { url: '/episode/some-slug' } } })
    expect(navigate).toHaveBeenCalledWith('/episode/some-slug')
  })

  it("opens What's new for a push about several episodes", async () => {
    const navigate = vi.fn()
    await initPushTaps(navigate)
    onAction!({ actionId: 'tap', notification: { data: { url: 'https://closelistening.app/#whats-new' } } })
    expect(navigate).toHaveBeenCalledWith('/#whats-new')
  })

  it('does nothing for a dismiss, or a push with no url', async () => {
    const navigate = vi.fn()
    await initPushTaps(navigate)
    onAction!({ actionId: 'dismiss', notification: { data: { url: '/episode/x' } } })
    onAction!({ actionId: 'tap', notification: { data: {} } })
    onAction!({ actionId: 'tap', notification: {} })
    expect(navigate).not.toHaveBeenCalled()
  })

  it('registers nothing on the web, where the service worker handles the click', async () => {
    native = false
    await initPushTaps(vi.fn())
    expect(PushNotifications.addListener).not.toHaveBeenCalled()
  })
})

describe('pushTapPath', () => {
  it('opens the in-app place a real (absolutised) push names', () => {
    // What the delivery worker actually sends: the url made absolute against the app origin.
    expect(pushTapPath('https://closelistening.app/#whats-new')).toBe('/#whats-new')
    expect(pushTapPath('https://closelistening.app/episode/a?t=30')).toBe('/episode/a?t=30')
    // The dev tenant's origin is different; the place is the same.
    expect(pushTapPath('http://100.1.2.3:8080/episode/a')).toBe('/episode/a')
    expect(pushTapPath('/episode/a')).toBe('/episode/a')
    expect(pushTapPath('/#whats-new')).toBe('/#whats-new')
  })

  it('never yields anything but an in-app path', () => {
    // A foreign host is reduced to a route in THIS app — router.push cannot leave it.
    expect(pushTapPath('//evil.example/x')).toBe('/x')
    expect(pushTapPath('javascript:alert(1)')).toBeNull()
    expect(pushTapPath('closelistening://episode/a')).toBeNull()
    expect(pushTapPath('')).toBeNull()
    expect(pushTapPath(42)).toBeNull()
    expect(pushTapPath(undefined)).toBeNull()
  })
})
