/**
 * Measure how the native app comes back: warm resume, WebView reload, or cold launch (#2277).
 *
 * The operator saw the app "start from scratch" several times a day after ~10 minutes away, and
 * nothing recorded it. GlitchTip cannot: Sentry reports watchdog terminations only when the app was
 * in the FOREGROUND, and a background eviction is not a crash. The device's Analytics Data had no
 * JetsamEvent either. So the app reports it itself:
 *
 *   - `app_launch` once per page boot — `kind` (did the whole process start with this page, or is the
 *     process older, i.e. only the WebView was reloaded), `previous_exit` (was the last recorded
 *     state background or foreground), `away` (how long since it went to the background) and
 *     `restored` (what #2278 brought back);
 *   - `app_resume` for a warm return to a live app, with the same `away`.
 *
 * Both screens look identical to the user — the web splash overlay shows on any page boot — which is
 * why `kind` needs the native process uptime rather than anything the page can see.
 */
import { App } from '@capacitor/app'
import { registerPlugin } from '@capacitor/core'
import { resolveSession, track, type AwayBucket, type LaunchKind } from './analytics'
import { postAppExits, type AppExitEntry } from './api'
import { getDeviceJson, setDeviceJson } from './deviceStore'
import { restoreOutcome } from './lastPlace'
import { isNative } from './native'

export const LIFECYCLE_KEY = 'lifecycle.last'

interface LastState {
  state: 'foreground' | 'background'
  at: number
}

interface AppProcessPlugin {
  uptime(): Promise<{ ms: number }>
  /** #2279: pending exit records, oldest first. Android adds the exit-history watermark. */
  exitLog(): Promise<{ entries: AppExitEntry[]; historyThrough?: number }>
  /** Clear what was delivered: iOS by `count`, Android by `historyThrough`. */
  clearExitLog(opts: { count: number; historyThrough?: number }): Promise<void>
}
const AppProcess = registerPlugin<AppProcessPlugin>('AppProcess')

/**
 * A page booted within this long of its process is a cold launch. Generous on purpose: the gap is
 * native start-up plus loading the bundle, seconds at most, while a reload lands minutes or hours
 * into a process that was suspended in the background.
 */
const COLD_GAP_MS = 15_000

export function classifyLaunch(processUptimeMs: number | null, pageAgeMs: number): LaunchKind {
  if (processUptimeMs === null || !Number.isFinite(processUptimeMs)) return 'unknown'
  return processUptimeMs - pageAgeMs < COLD_GAP_MS ? 'cold' : 'webview_reload'
}

export function toAwayBucket(ms: number | null): AwayBucket {
  if (ms === null || !Number.isFinite(ms) || ms < 0) return 'none'
  const min = ms / 60_000
  if (min < 1) return '<1m'
  if (min < 5) return '1-5m'
  if (min < 15) return '5-15m'
  if (min < 60) return '15-60m'
  return '1h+'
}

/**
 * Forward why the app last ended to the server, then clear what was delivered (#2279).
 *
 * The device records it natively — MetricKit's daily exit counts and memory warnings on iOS, the
 * system's exit history on Android, and the WebView's content/renderer process dying on both —
 * because the web layer is exactly what is gone when it happens. Cleared only after the server
 * accepted it, so an offline launch keeps the records for the next one.
 */
export async function forwardExitLog(): Promise<void> {
  try {
    const { entries, historyThrough } = await AppProcess.exitLog()
    if (!entries?.length) return
    const session = resolveSession()
    if (session.platform === 'web') return
    const ok = await postAppExits({
      platform: session.platform,
      app_version: session.app_version,
      entries: entries.slice(0, 50),
    })
    if (ok) await AppProcess.clearExitLog({ count: Math.min(entries.length, 50), historyThrough })
  } catch {
    // Telemetry never breaks the app.
  }
}

async function writeState(state: LastState['state'], at = Date.now()): Promise<void> {
  try {
    await setDeviceJson(LIFECYCLE_KEY, { state, at } satisfies LastState)
  } catch {
    // A lost write mislabels one launch; it must never break one.
  }
}

/**
 * Report this launch and watch the app's state from here on. Native only. Call once, after the
 * router's first navigation, so `restored` is known.
 *
 * `onBackground` runs as the app leaves the foreground — the last reliable moment to save anything.
 */
export async function initLifecycle(onBackground: () => Promise<void> | void): Promise<void> {
  if (!isNative()) return
  try {
    const pageAge = performance.now()
    const previous = await getDeviceJson<LastState>(LIFECYCLE_KEY).catch(() => null)
    const uptime = await AppProcess.uptime()
      .then((r) => r.ms)
      .catch(() => null)
    const now = Date.now()
    track('app_launch', {
      kind: classifyLaunch(uptime, pageAge),
      previous_exit: previous ? previous.state : 'first',
      away: toAwayBucket(previous?.state === 'background' ? now - previous.at : null),
      restored: restoreOutcome(),
    })
    await writeState('foreground', now)
    void forwardExitLog()

    let backgroundAt: number | null = null
    await App.addListener('appStateChange', ({ isActive }) => {
      if (!isActive) {
        backgroundAt = Date.now()
        void writeState('background', backgroundAt)
        void onBackground()
        return
      }
      if (backgroundAt !== null) {
        track('app_resume', { away: toAwayBucket(Date.now() - backgroundAt) })
        backgroundAt = null
      }
      void writeState('foreground')
    })
  } catch {
    // Telemetry never breaks the app.
  }
}
