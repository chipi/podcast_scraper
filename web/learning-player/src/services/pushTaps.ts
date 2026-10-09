import { Capacitor } from '@capacitor/core'
import { PushNotifications } from '@capacitor/push-notifications'

/**
 * Open where a tapped push points (2026-10-09).
 *
 * The delivery template puts the target in `url` — the episode for one new episode, Home's
 * What's new (`/#whats-new`) for several. APNs carries it as a custom key and FCM in `data`, and
 * the plugin surfaces both as `notification.data.url`. Before this, a tap on the native app only
 * opened it wherever it was; the web service worker (push-sw.js) already navigated.
 *
 * Both plugins retain the event until a listener exists, so a tap that LAUNCHED the app is
 * delivered once this registers. `navigate` is injected for the same reason as initDeepLinks: the
 * shell owns the router.
 */
export function pushTapPath(url: unknown): string | null {
  if (typeof url !== 'string' || !url) return null
  // The delivery worker ABSOLUTISES the url against the tenant's app origin (closelistening.app on
  // prod, the dev origin on dev), so a real push says "https://closelistening.app/#whats-new". The
  // app opens its own copy of that place: path, query and hash, whatever the host — the result is
  // only ever an in-app route, so it cannot take the reader out of the app.
  let parsed: URL
  try {
    parsed = new URL(url, 'https://app.invalid')
  } catch {
    return null
  }
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') return null
  return `${parsed.pathname}${parsed.search}${parsed.hash}`
}

export async function initPushTaps(navigate: (path: string) => void): Promise<void> {
  if (!Capacitor.isNativePlatform()) return
  await PushNotifications.addListener('pushNotificationActionPerformed', (action) => {
    if (action.actionId !== 'tap') return
    const path = pushTapPath(action.notification?.data?.url)
    if (path) navigate(path)
  })
}
