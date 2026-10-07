import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * Native push registration — what `kind` a device token is stored under (#2157).
 *
 * This is the field the delivery worker's dispatcher routes on. Android device tokens are FCM
 * tokens and were being written as `kind: 'apns'`; nothing read `kind` yet, so it was inert rather
 * than broken — but the moment routing lands, a store full of mislabelled tokens sends every
 * Android push to Apple. The label is therefore worth a test of its own, ahead of the transport.
 */

let platform = 'ios'
vi.mock('@capacitor/core', () => ({
  Capacitor: { isNativePlatform: () => true, getPlatform: () => platform },
}))

type RegListener = (t: { value: string }) => void | Promise<void>
const listeners: Record<string, RegListener> = {}
// What the OS hands back on the next register(), and what checkPermissions() reports.
let nextToken = 'TOKEN123'
let permission = 'granted'
const register = vi.fn(async () => {
  // The plugin delivers the token asynchronously; mimic that rather than resolving inline.
  await Promise.resolve()
  await listeners.registration?.({ value: nextToken })
})
vi.mock('@capacitor/push-notifications', () => ({
  PushNotifications: {
    requestPermissions: vi.fn(async () => ({ receive: 'granted' })),
    checkPermissions: vi.fn(async () => ({ receive: permission })),
    register: () => register(),
    unregister: vi.fn(async () => undefined),
    addListener: vi.fn(async (event: string, cb: RegListener) => {
      listeners[event] = cb
      return { remove: vi.fn() }
    }),
  },
}))

const prefs: Record<string, string> = {}
vi.mock('@capacitor/preferences', () => ({
  Preferences: {
    set: vi.fn(async ({ key, value }: { key: string; value: string }) => {
      prefs[key] = value
    }),
    get: vi.fn(async ({ key }: { key: string }) => ({ value: prefs[key] ?? null })),
    remove: vi.fn(async ({ key }: { key: string }) => {
      delete prefs[key]
    }),
  },
}))

// Typed with its argument: the module forwards the subscription, and the assertions read it back
// from `mock.calls[0][0]` — a zero-arg mock made both of those a type error under vue-tsc.
const subscribePush = vi.fn(async (_subscription: unknown) => ({ count: 1 }))
const unsubscribePush = vi.fn(async (_endpoint: string) => undefined)
vi.mock('../services/api', () => ({
  subscribePush: (s: unknown) => subscribePush(s as never),
  unsubscribePush: (e: string) => unsubscribePush(e),
  getVapidKey: vi.fn(async () => ''),
}))

vi.mock('../services/native', () => ({ isNative: () => true }))

describe('native push registration', () => {
  beforeEach(() => {
    subscribePush.mockClear()
    subscribePush.mockImplementation(async () => ({ count: 1 }))
    unsubscribePush.mockClear()
    register.mockClear()
    nextToken = 'TOKEN123'
    permission = 'granted'
    for (const k of Object.keys(prefs)) delete prefs[k]
  })
  afterEach(() => {
    vi.resetModules()
  })

  it('stores an iOS token as apns, with a matching endpoint scheme', async () => {
    platform = 'ios'
    const { enablePush } = await import('./usePushSubscription')
    await expect(enablePush()).resolves.toBe(true)

    expect(subscribePush).toHaveBeenCalledTimes(1)
    const sent = subscribePush.mock.calls[0][0] as unknown as Record<string, unknown>
    expect(sent.kind).toBe('apns')
    expect(sent.endpoint).toBe('apns://TOKEN123')
    expect(sent.token).toBe('TOKEN123')
  })

  it('labels an Android token fcm, and an iOS one apns', async () => {
    // Tested directly rather than through `enablePush`, because Android cannot reach the
    // registration path at all while the guard below holds — so an end-to-end assertion here
    // would be testing the guard and silently claiming to test the label.
    vi.resetModules()
    const { nativePushKind } = await import('./usePushSubscription')
    platform = 'android'
    expect(nativePushKind()).toBe('fcm')
    platform = 'ios'
    expect(nativePushKind()).toBe('apns')
  })

  it('registers on Android now that Firebase and a sender exist (#2157)', async () => {
    // ANDROID_PUSH_NATIVE_READY was false while a missing google-services.json would make the
    // plugin throw `IllegalStateException` on a native handler thread — uncatchable from JS, so it
    // killed the process. Firebase config now ships in the Android build, so the path is open and
    // this is the end-to-end assertion the guard previously made impossible.
    platform = 'android'
    vi.resetModules()
    const mod = await import('./usePushSubscription')

    expect(mod.pushSupported()).toBe(true)
    await expect(mod.enablePush()).resolves.toBe(true)

    const sent = subscribePush.mock.calls[0][0] as unknown as Record<string, unknown>
    expect(sent.kind).toBe('fcm')
    expect(sent.endpoint).toBe('fcm://TOKEN123')
    expect(sent.platform).toBe('android')
  })

  it('iOS is supported natively without any guard', async () => {
    platform = 'ios'
    vi.resetModules()
    const { pushSupported } = await import('./usePushSubscription')
    expect(pushSupported()).toBe(true)
  })

  // Prod 2026-10-07: one phone held two FCM tokens — the OS had issued a new one, the app added it
  // beside the old, and the old stayed until a send bounced on it (404 UNREGISTERED).
  it('a new token retires the one this device registered before', async () => {
    platform = 'android'
    prefs['push.apnsEndpoint'] = 'fcm://OLD'
    nextToken = 'NEW'
    const { enablePush } = await import('./usePushSubscription')
    await expect(enablePush()).resolves.toBe(true)

    expect((subscribePush.mock.calls[0][0] as { endpoint: string }).endpoint).toBe('fcm://NEW')
    expect(unsubscribePush).toHaveBeenCalledWith('fcm://OLD')
    expect(prefs['push.apnsEndpoint']).toBe('fcm://NEW')
    // New stored BEFORE the old is retired: never a moment with nothing on the server.
    expect(subscribePush.mock.invocationCallOrder[0]).toBeLessThan(
      unsubscribePush.mock.invocationCallOrder[0],
    )
  })

  it('the same token retires nothing', async () => {
    platform = 'ios'
    prefs['push.apnsEndpoint'] = 'apns://TOKEN123'
    const { enablePush } = await import('./usePushSubscription')
    await expect(enablePush()).resolves.toBe(true)
    expect(unsubscribePush).not.toHaveBeenCalled()
  })

  it('a failed registration keeps the old token, on the server and on the device', async () => {
    platform = 'android'
    prefs['push.apnsEndpoint'] = 'fcm://OLD'
    nextToken = 'NEW'
    subscribePush.mockImplementation(async () => {
      throw new Error('offline')
    })
    const { enablePush } = await import('./usePushSubscription')
    await expect(enablePush()).resolves.toBe(false)
    expect(unsubscribePush).not.toHaveBeenCalled()
    expect(prefs['push.apnsEndpoint']).toBe('fcm://OLD')
  })

  describe('refreshNativePushToken (at launch)', () => {
    it('re-registers a rotated token and retires the old one', async () => {
      platform = 'android'
      prefs['push.apnsEndpoint'] = 'fcm://OLD'
      nextToken = 'ROTATED'
      const { refreshNativePushToken } = await import('./usePushSubscription')
      await expect(refreshNativePushToken()).resolves.toBe(true)
      expect((subscribePush.mock.calls[0][0] as { endpoint: string }).endpoint).toBe(
        'fcm://ROTATED',
      )
      expect(unsubscribePush).toHaveBeenCalledWith('fcm://OLD')
    })

    it('does nothing on a device that never turned push on', async () => {
      platform = 'ios'
      const { refreshNativePushToken } = await import('./usePushSubscription')
      await expect(refreshNativePushToken()).resolves.toBe(false)
      expect(register).not.toHaveBeenCalled()
      expect(subscribePush).not.toHaveBeenCalled()
    })

    it('never prompts: no granted permission, no registration', async () => {
      platform = 'ios'
      prefs['push.apnsEndpoint'] = 'apns://OLD'
      permission = 'prompt'
      const mod = await import('./usePushSubscription')
      const { PushNotifications } = await import('@capacitor/push-notifications')
      vi.mocked(PushNotifications.requestPermissions).mockClear()
      await expect(mod.refreshNativePushToken()).resolves.toBe(false)
      expect(register).not.toHaveBeenCalled()
      expect(PushNotifications.requestPermissions).not.toHaveBeenCalled()
    })
  })
})
