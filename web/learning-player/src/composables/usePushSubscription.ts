/**
 * Push subscription (PRD-046 FR1 / #1415). Two transports behind one API, chosen by platform:
 *
 *  - **Native (iOS/Android Capacitor app):** APNs/FCM via `@capacitor/push-notifications`. The
 *    WKWebView has no Service Worker / `PushManager`, so Web Push cannot work in the app — the plugin
 *    registers with the OS, hands back a device token, and we store it server-side as an `apns`
 *    subscription. This is why "enable push" now works in the iOS app (operator 2026-09-14).
 *  - **Web (PWA / browser):** the W3C Push API + VAPID, as before.
 *
 * `enablePush` returns false when push can't be enabled (permission denied / plugin error / not
 * configured server-side) so the UI can revert the toggle.
 */
import { Capacitor, type PluginListenerHandle } from '@capacitor/core'
import { PushNotifications } from '@capacitor/push-notifications'
import { Preferences } from '@capacitor/preferences'
import { getVapidKey, subscribePush, unsubscribePush } from '../services/api'
import { isNative } from '../services/native'

// Where we remember THIS device's APNs endpoint, so a later "disable" can deregister it server-side
// (the token is not otherwise recoverable without re-registering).
const APNS_ENDPOINT_KEY = 'push.apnsEndpoint'
const REGISTER_TIMEOUT_MS = 15_000

/** Whether this platform can do push at all. Native always can (the OS owns it); web needs the APIs. */
export function pushSupported(): boolean {
  if (isNative()) return true
  return (
    typeof navigator !== 'undefined' &&
    'serviceWorker' in navigator &&
    typeof window !== 'undefined' &&
    'PushManager' in window &&
    'Notification' in window
  )
}

// VAPID keys travel as URL-safe base64; the browser wants a Uint8Array applicationServerKey.
function urlBase64ToUint8Array(base64: string): Uint8Array {
  const padding = '='.repeat((4 - (base64.length % 4)) % 4)
  const normalized = (base64 + padding).replace(/-/g, '+').replace(/_/g, '/')
  const raw = atob(normalized)
  const out = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i += 1) out[i] = raw.charCodeAt(i)
  return out
}

/**
 * Native: ask the OS, register for a device token, store it server-side as an `apns` subscription.
 * The token arrives asynchronously on the `registration` event; resolve false on error/timeout so
 * the toggle reverts rather than hanging.
 */
async function enablePushNative(): Promise<boolean> {
  const perm = await PushNotifications.requestPermissions()
  if (perm.receive !== 'granted') return false

  return await new Promise<boolean>((resolve) => {
    let settled = false
    let regHandle: PluginListenerHandle | undefined
    let errHandle: PluginListenerHandle | undefined
    const finish = (ok: boolean): void => {
      if (settled) return
      settled = true
      void regHandle?.remove()
      void errHandle?.remove()
      resolve(ok)
    }
    void PushNotifications.addListener('registration', async (t) => {
      const token = t.value
      const endpoint = `apns://${token}`
      try {
        await subscribePush({ endpoint, kind: 'apns', platform: Capacitor.getPlatform(), token })
        await Preferences.set({ key: APNS_ENDPOINT_KEY, value: endpoint })
        finish(true)
      } catch {
        finish(false)
      }
    }).then((h) => {
      regHandle = h
    })
    void PushNotifications.addListener('registrationError', () => finish(false)).then((h) => {
      errHandle = h
    })
    void PushNotifications.register()
    setTimeout(() => finish(false), REGISTER_TIMEOUT_MS)
  })
}

async function disablePushNative(): Promise<void> {
  await PushNotifications.unregister().catch(() => undefined)
  const { value } = await Preferences.get({ key: APNS_ENDPOINT_KEY })
  if (value) {
    await unsubscribePush(value).catch(() => undefined)
    await Preferences.remove({ key: APNS_ENDPOINT_KEY })
  }
}

/** Subscribe this device/browser + register with the server. Returns false if push can't be enabled. */
export async function enablePush(): Promise<boolean> {
  if (isNative()) return enablePushNative()
  if (!pushSupported()) return false
  const permission = await Notification.requestPermission()
  if (permission !== 'granted') return false
  let key: string
  try {
    key = await getVapidKey()
  } catch {
    return false // push not configured server-side (503)
  }
  if (!key) return false
  const registration = await navigator.serviceWorker.ready
  const subscription = await registration.pushManager.subscribe({
    userVisibleOnly: true,
    // Runtime value is a valid BufferSource; the cast sidesteps lib.dom's ArrayBufferLike narrowing.
    applicationServerKey: urlBase64ToUint8Array(key) as BufferSource,
  })
  await subscribePush(subscription.toJSON())
  return true
}

/** Unsubscribe this device/browser + deregister with the server. Safe to call when not subscribed. */
export async function disablePush(): Promise<void> {
  if (isNative()) {
    await disablePushNative()
    return
  }
  if (!pushSupported()) return
  const registration = await navigator.serviceWorker.ready
  const subscription = await registration.pushManager.getSubscription()
  if (!subscription) return
  await unsubscribePush(subscription.endpoint).catch(() => undefined)
  await subscription.unsubscribe().catch(() => undefined)
}
