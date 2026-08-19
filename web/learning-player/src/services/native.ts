/**
 * Native-shell helpers (#1310). One place for Capacitor platform detection + the native
 * implementations of things that behave differently in the iOS/Android WebView than on the web
 * (file download/share; later: OAuth, background audio). Everything here is guarded so the web
 * build is unaffected — `isNative()` is false on the web and callers take the browser path.
 *
 * Capacitor's plugin JS is a thin bridge that no-ops to a rejected promise off-device, so importing
 * these at module top-level is safe on the web; we still gate on isNative() before calling them.
 */
import { App } from '@capacitor/app'
import { Browser } from '@capacitor/browser'
import { Capacitor, CapacitorCookies, registerPlugin } from '@capacitor/core'
import { Directory, Encoding, Filesystem } from '@capacitor/filesystem'
import { Share } from '@capacitor/share'
import { setAuthToken } from './api'
import { resolveApiBase, resolveGateCookie } from './tier'

// Local Android plugin (#1310): a foreground media service keep-alive so the OS doesn't suspend the
// WebView's <audio> when backgrounded. iOS handles background audio via AVAudioSession (AppDelegate)
// + UIBackgroundModes, so this is Android-only; the calls no-op elsewhere.
interface BackgroundAudioPlugin {
  start(): Promise<void>
  stop(): Promise<void>
}
const BackgroundAudio = registerPlugin<BackgroundAudioPlugin>('BackgroundAudio')

/** Start the Android foreground keep-alive (on play). No-op on iOS/web. */
export async function startBackgroundAudio(): Promise<void> {
  if (isNative() && platform() === 'android') await BackgroundAudio.start().catch(() => {})
}

/** Stop the Android foreground keep-alive (on pause/end). No-op on iOS/web. */
export async function stopBackgroundAudio(): Promise<void> {
  if (isNative() && platform() === 'android') await BackgroundAudio.stop().catch(() => {})
}

/** True inside the iOS/Android Capacitor shell; false on the web (SSR/dev/preview/prod web). */
export function isNative(): boolean {
  return Capacitor.isNativePlatform()
}

/** 'ios' | 'android' | 'web' — used for telemetry tagging + platform-specific branches. */
export function platform(): string {
  return Capacitor.getPlatform()
}

/**
 * Seed the pre-launch coming-soon gate's `cl_preview` cookie into the native cookie jar (prod tier,
 * internal builds only). With CapacitorHttp, native API requests to closelistening.app then carry it,
 * so caddy's @preview_ok opens the gate on the COOKIE — checked before @authed — and the
 * `Authorization` header is left free for the signed-in user's Bearer (fixes the Basic-vs-Bearer
 * collision that otherwise 401s every call once logged in). Call once before mount. No-op on web /
 * dev tier / release (resolveGateCookie returns null). Silent-degrade: the Basic-auth header fallback
 * (services/api.ts) still covers anonymous reads if this fails.
 */
export async function initGateCookie(): Promise<void> {
  const value = resolveGateCookie()
  if (!value) return
  try {
    const origin = new URL(resolveApiBase(), 'https://closelistening.app').origin
    await CapacitorCookies.setCookie({ url: origin, key: 'cl_preview', value })
  } catch {
    /* jar unavailable — Basic-auth header still clears the gate for anonymous reads */
  }
}

// --- native OAuth (#1310) -------------------------------------------------------------------------
// The session cookie can't cross an external OAuth browser into the WebView, so the backend hands
// back the SAME signed token via a `closelistening://auth#token=...` callback. We persist it in
// localStorage (durable in WKWebView across launches; re-login if the OS evicts it) and set it as
// the bearer token used by services/api.ts. Kept out of Preferences to avoid another plugin dep.
//
// Return mechanism differs by platform so there is NO "Open in app?" dialog:
//   - iOS: ASWebAuthenticationSession (local AuthSession plugin) — Apple's OAuth primitive returns
//     the callback URL directly to a promise; no custom-scheme confirmation, no deep-link listener.
//   - Android: @capacitor/browser + the manifest intent-filter delivers the callback via appUrlOpen
//     with no prompt.
const CALLBACK_SCHEME = 'closelistening'
const TOKEN_KEY = 'lp_native_token'

// Local iOS plugin: wraps ASWebAuthenticationSession (see ios/App/App/AuthSession.swift).
interface AuthSessionPlugin {
  start(opts: { url: string; callbackScheme: string }): Promise<{ url: string }>
}
const AuthSession = registerPlugin<AuthSessionPlugin>('AuthSession')

// After a token arrives (either platform), refresh the auth store. Set by initNativeAuth().
let onAuthedCb: (() => void) | null = null

/** Pull the signed token out of a `closelistening://auth#token=<signed>` callback URL. */
function tokenFromCallback(url: string): string | null {
  const frag = url.includes('#') ? url.slice(url.indexOf('#') + 1) : ''
  return new URLSearchParams(frag).get('token')
}

/** Persist (or clear) the native bearer token + apply it to the API client. */
export function storeAuthToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* private-mode / storage disabled — the in-memory token below still works for this session */
  }
  setAuthToken(token)
}

/**
 * Begin native OAuth (#1310). iOS uses ASWebAuthenticationSession (prompt-free, returns the callback
 * URL directly); Android opens the system browser and the appUrlOpen listener (initNativeAuth)
 * receives the intent-filter callback. Both end by storing the token + refreshing the auth store.
 */
export async function startNativeLogin(loginUrl: string): Promise<void> {
  if (!isNative()) return
  if (platform() === 'ios') {
    try {
      const { url } = await AuthSession.start({ url: loginUrl, callbackScheme: CALLBACK_SCHEME })
      const token = tokenFromCallback(url)
      if (token) {
        storeAuthToken(token)
        onAuthedCb?.()
      }
    } catch {
      /* user cancelled or the session failed — stay signed out */
    }
    return
  }
  // Android: the manifest intent-filter routes the callback back via appUrlOpen (no prompt).
  await Browser.open({ url: loginUrl })
}

/**
 * Rehydrate the stored token and register the Android deep-link handler. Call once at startup
 * (App.vue). `onAuthed` runs after a token arrives (either platform) so the app can refresh the
 * auth store. No-op on the web.
 */
export async function initNativeAuth(onAuthed: () => void): Promise<void> {
  if (!isNative()) return
  onAuthedCb = onAuthed
  try {
    const saved = localStorage.getItem(TOKEN_KEY)
    if (saved) setAuthToken(saved)
  } catch {
    /* ignore */
  }
  // Android callback path (iOS returns via the AuthSession promise instead).
  await App.addListener('appUrlOpen', ({ url }) => {
    const token = tokenFromCallback(url)
    if (token) {
      storeAuthToken(token)
      void Browser.close().catch(() => {})
      onAuthed()
    }
  })
}

/**
 * Save text to a file and hand it to the OS share sheet (native replacement for `<a download>`,
 * which doesn't save in WKWebView). Writes to the Cache dir (transient, no permission prompt) then
 * shares the resulting file URI so the user can save/send it wherever they want.
 */
export async function saveAndShareText(
  filename: string,
  text: string,
  mimeType = 'text/markdown',
): Promise<void> {
  const { uri } = await Filesystem.writeFile({
    path: filename,
    data: text,
    directory: Directory.Cache,
    encoding: Encoding.UTF8,
  })
  await Share.share({ title: filename, url: uri, dialogTitle: filename })
  // Best-effort cleanup — the share is synchronous with the sheet; leave the file for the OS to
  // reap from Cache (deleting immediately can race the receiving app reading the URI).
  void mimeType
}
