/**
 * Push subscription (PRD-046 FR1 / #1415). Two transports behind one API, chosen by platform:
 *
 *  - **Native (iOS/Android Capacitor app):** APNs/FCM via `@capacitor/push-notifications`. The
 *    WKWebView has no Service Worker / `PushManager`, so Web Push cannot work in the app — the plugin
 *    registers with the OS, hands back a device token, and we store it server-side as an `apns`
 *    (iOS) or `fcm` (Android) subscription; see `nativePushKind`. Android is gated off here
 *    entirely; see `ANDROID_PUSH_NATIVE_READY` (#2157).
 *  - **Web (PWA / browser):** the W3C Push API + VAPID, as before.
 *
 * `enablePush` returns false when push can't be enabled (permission denied / plugin error / platform
 * not stood up / not configured server-side) so the UI can revert the toggle.
 *
 * SCOPE: this registers a device token; it does not mean a notification arrives. No device-token
 * sender exists server-side — the store keeps W3C subscriptions and the worker signs with VAPID,
 * which cannot reach APNs or FCM. Treat native push as unverified end-to-end until #2157 closes.
 */
import { Capacitor, type PluginListenerHandle } from '@capacitor/core'
import { PushNotifications } from '@capacitor/push-notifications'
import { Preferences } from '@capacitor/preferences'
import { getVapidKey, subscribePush, unsubscribePush } from '../services/api'
import { isNative } from '../services/native'

// Where we remember THIS device's native push endpoint, so a later "disable" can deregister it
// server-side (the token is not otherwise recoverable without re-registering).
//
// The key still says "apns" although it now holds an FCM endpoint on Android too. Renaming it
// would strand every endpoint an already-installed iOS app has stored — that device could never
// deregister itself again — which is a real cost for a cosmetic gain.
const NATIVE_ENDPOINT_KEY = 'push.apnsEndpoint'
const REGISTER_TIMEOUT_MS = 15_000

/**
 * What transport this platform's device token belongs to.
 *
 * Android device tokens are **FCM** tokens, and were being sent as `kind: 'apns'` with an
 * `apns://` endpoint (#2157). Nothing reads `kind` server-side yet, so this was inert rather than
 * broken — but it is the field the dispatcher must route on, and a store full of FCM tokens
 * labelled `apns` would send every Android push to Apple the moment routing lands.
 */
export function nativePushKind(): 'apns' | 'fcm' {
  return Capacitor.getPlatform() === 'android' ? 'fcm' : 'apns'
}

/**
 * Android push is not stood up yet (#2157), so we must not talk to the plugin there.
 *
 * Android push means FCM, and `google-services.json` is gitignored by design, so no build carries
 * Firebase config. Calling into the plugin without it raises `IllegalStateException: Default
 * FirebaseApp is not initialized` on a native handler thread — which JS cannot catch, so it kills
 * the process rather than rejecting a promise. Gating here rather than at the `register()` call is
 * deliberate: it also covers `requestPermissions()` and `unregister()`, so the guard does not
 * depend on knowing which plugin method throws first.
 *
 * Flip to true once #2157 lands the Firebase config AND an FCM sender server-side; delete the
 * constant once delivery is verified on a real device.
 *
 * FLIP `-PandroidPushRequired=true` IN THE SAME CHANGE (see android/app/build.gradle). Until then
 * a release build with no `google-services.json` only warns, because a push-less Android build is
 * the expected state while this is false. The moment it is true, that same missing file becomes an
 * artifact that crashes the first time a tester touches the notifications toggle — and the gradle
 * gate is what stops it reaching them.
 */
const ANDROID_PUSH_NATIVE_READY = false

/** Whether this platform can do push at all. Native owns it via the OS; web needs the APIs. */
export function pushSupported(): boolean {
  if (isNative()) {
    return Capacitor.getPlatform() !== 'android' || ANDROID_PUSH_NATIVE_READY
  }
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
 * Native: ask the OS, register for a device token, store it server-side under the platform's own
 * transport (`apns` on iOS, `fcm` on Android). The token arrives asynchronously on the
 * `registration` event; resolve false on error/timeout so the toggle reverts rather than hanging.
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
      const kind = nativePushKind()
      // The scheme matches the kind, so an endpoint is self-describing in the store and in a log
      // line — `apns://…` for Apple, `fcm://…` for Google.
      const endpoint = `${kind}://${token}`
      try {
        await subscribePush({ endpoint, kind, platform: Capacitor.getPlatform(), token })
        await Preferences.set({ key: NATIVE_ENDPOINT_KEY, value: endpoint })
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
  const { value } = await Preferences.get({ key: NATIVE_ENDPOINT_KEY })
  if (value) {
    await unsubscribePush(value).catch(() => undefined)
    await Preferences.remove({ key: NATIVE_ENDPOINT_KEY })
  }
}

/** Subscribe this device/browser + register with the server. Returns false if push can't be enabled. */
export async function enablePush(): Promise<boolean> {
  if (isNative()) return pushSupported() ? enablePushNative() : false
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
    // Nothing was ever subscribed where push is unsupported, so skipping the plugin loses nothing
    // — and `unregister()` deletes the FCM token, which is the second way to hit the #2157 crash.
    if (pushSupported()) await disablePushNative()
    return
  }
  if (!pushSupported()) return
  const registration = await navigator.serviceWorker.ready
  const subscription = await registration.pushManager.getSubscription()
  if (!subscription) return
  await unsubscribePush(subscription.endpoint).catch(() => undefined)
  await subscription.unsubscribe().catch(() => undefined)
}
