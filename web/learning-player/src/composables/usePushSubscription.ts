/**
 * Web Push subscription (PRD-046 FR1 / #1415). Wraps the browser Push API + the app's
 * push endpoints so ProfileView can flip the "push reminders" toggle. Degrades gracefully:
 * `enablePush` returns false when the browser can't do push (no SW / no PushManager / denied
 * permission / push not configured server-side), so the UI can revert the toggle.
 */
import { getVapidKey, subscribePush, unsubscribePush } from '../services/api'

/** Whether this browser can do Web Push at all. */
export function pushSupported(): boolean {
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

/** Subscribe this browser + register with the server. Returns false if push can't be enabled. */
export async function enablePush(): Promise<boolean> {
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

/** Unsubscribe this browser + deregister with the server. Safe to call when not subscribed. */
export async function disablePush(): Promise<void> {
  if (!pushSupported()) return
  const registration = await navigator.serviceWorker.ready
  const subscription = await registration.pushManager.getSubscription()
  if (!subscription) return
  await unsubscribePush(subscription.endpoint).catch(() => undefined)
  await subscription.unsubscribe().catch(() => undefined)
}
