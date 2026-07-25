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
import { Capacitor, registerPlugin } from '@capacitor/core'
import { Directory, Encoding, Filesystem } from '@capacitor/filesystem'
import { Share } from '@capacitor/share'
import { setAuthToken } from './api'

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

// --- native OAuth (#1310) -------------------------------------------------------------------------
// The session cookie can't cross an external OAuth browser into the WebView, so the backend hands
// back the SAME signed token via a `closelistening://auth#token=...` deep link. We persist it in
// localStorage (durable in WKWebView across launches; re-login if the OS evicts it) and set it as
// the bearer token used by services/api.ts. Kept out of Preferences to avoid another plugin dep.
const TOKEN_KEY = 'lp_native_token'

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

/** Open the OAuth login URL in the system browser (native). Web callers use a full-page redirect. */
export async function openOAuth(url: string): Promise<void> {
  await Browser.open({ url })
}

/**
 * Rehydrate the stored token and register the deep-link handler that receives the OAuth callback.
 * Idempotent-ish: call once at startup (App.vue). `onAuthed` runs after a token arrives so the app
 * can refresh the auth store. No-op on the web.
 */
export async function initNativeAuth(onAuthed: () => void): Promise<void> {
  if (!isNative()) return
  try {
    const saved = localStorage.getItem(TOKEN_KEY)
    if (saved) setAuthToken(saved)
  } catch {
    /* ignore */
  }
  await App.addListener('appUrlOpen', ({ url }) => {
    // Expect closelistening://auth#token=<signed> (token in the fragment, not the query).
    const frag = url.includes('#') ? url.slice(url.indexOf('#') + 1) : ''
    const token = new URLSearchParams(frag).get('token')
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
