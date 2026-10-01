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
vi.mock('@capacitor/push-notifications', () => ({
  PushNotifications: {
    requestPermissions: vi.fn(async () => ({ receive: 'granted' })),
    register: vi.fn(async () => {
      // The plugin delivers the token asynchronously; mimic that rather than resolving inline.
      await Promise.resolve()
      await listeners.registration?.({ value: 'TOKEN123' })
    }),
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
vi.mock('../services/api', () => ({
  subscribePush: (s: unknown) => subscribePush(s as never),
  unsubscribePush: vi.fn(async () => undefined),
  getVapidKey: vi.fn(async () => ''),
}))

vi.mock('../services/native', () => ({ isNative: () => true }))

describe('native push registration', () => {
  beforeEach(() => {
    subscribePush.mockClear()
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
})
