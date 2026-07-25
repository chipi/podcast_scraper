/**
 * Native-shell helpers (#1310). One place for Capacitor platform detection + the native
 * implementations of things that behave differently in the iOS/Android WebView than on the web
 * (file download/share; later: OAuth, background audio). Everything here is guarded so the web
 * build is unaffected — `isNative()` is false on the web and callers take the browser path.
 *
 * Capacitor's plugin JS is a thin bridge that no-ops to a rejected promise off-device, so importing
 * these at module top-level is safe on the web; we still gate on isNative() before calling them.
 */
import { Capacitor } from '@capacitor/core'
import { Directory, Encoding, Filesystem } from '@capacitor/filesystem'
import { Share } from '@capacitor/share'

/** True inside the iOS/Android Capacitor shell; false on the web (SSR/dev/preview/prod web). */
export function isNative(): boolean {
  return Capacitor.isNativePlatform()
}

/** 'ios' | 'android' | 'web' — used for telemetry tagging + platform-specific branches. */
export function platform(): string {
  return Capacitor.getPlatform()
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
