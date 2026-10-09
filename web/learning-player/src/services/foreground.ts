/**
 * Run something each time the app comes back to the FOREGROUND (2026-10-09).
 *
 * Why: a push ("3 new episodes are out") is tapped while the app is still alive in the background,
 * so opening it is a RESUME, not a launch — nothing that loads "on sign-in" runs again. The bell
 * kept the inbox it had and showed the new alerts only a minute or two later (operator, on device).
 *
 * Native: Capacitor's App `resume` (what the OS reports when the app returns to the screen — a push
 * tap included). Web: `visibilitychange` to visible, the browser's equivalent for a tab or a PWA.
 * Returns an unsubscribe.
 */
import { App } from '@capacitor/app'
import { isNativeShell } from './tier'

export function onForeground(run: () => void): () => void {
  if (isNativeShell()) {
    const handle = App.addListener('resume', run)
    return () => void handle.then((h) => h.remove()).catch(() => {})
  }
  const onVisible = (): void => {
    if (document.visibilityState === 'visible') run()
  }
  document.addEventListener('visibilitychange', onVisible)
  return () => document.removeEventListener('visibilitychange', onVisible)
}
