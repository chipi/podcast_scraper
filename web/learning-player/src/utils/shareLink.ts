/**
 * The link a listener SHARES — one https URL for everything, the way YouTube and Spotify do it
 * (operator 2026-10-05): the recipient's phone opens it in the app when the app is installed
 * (Universal Links / App Links), the browser otherwise, and either one sends a signed-out visitor
 * through sign-in and back to the thing the link named.
 *
 * Built from the page's own origin on the web — the right host on every tier — and from the public
 * site inside the native shells. There the page's origin is the WebView's private one
 * (`capacitor://localhost` on iOS, `https://localhost` on Android): every link copied from a phone
 * pointed at the recipient's own device and opened nothing.
 */
import { isNativeShell } from '../services/tier'

export const PUBLIC_ORIGIN = 'https://closelistening.app'

export type ShareTarget = 'episode' | 'podcast' | 'topic' | 'person' | 'storyline' | 'theme'

export function shareOrigin(): string {
  if (!isNativeShell() && typeof window !== 'undefined' && window.location?.origin) {
    return window.location.origin
  }
  return PUBLIC_ORIGIN
}

/** `https://<origin>/<target>/<id>[?t=<seconds>]`. The id is encoded: graph ids carry a colon. */
export function shareUrl(target: ShareTarget, id: string, atSeconds?: number | null): string {
  const t = atSeconds != null && Number.isFinite(atSeconds) && atSeconds >= 0 ? `?t=${Math.floor(atSeconds)}` : ''
  return `${shareOrigin()}/${target}/${encodeURIComponent(id)}${t}`
}
