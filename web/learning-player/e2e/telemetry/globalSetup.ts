import { existsSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'
import { buildIndexIfAbsent } from '../globalSetup'

/**
 * Wipe the telemetry tier's per-user state before the run.
 *
 * Same class of leak the main config's `globalSetup` exists for, and it bit this tier on its first
 * full run. The specs sign in as STABLE identities (`telemetry-player`, `telemetry-follow`, …) so a
 * failure names the flow that broke; `APP_DATA_DIR` is gitignored but persists across local runs. So
 * the queue_add spec added an episode, and on the NEXT run that episode was already queued: the
 * control read "Remove from queue" instead of "Add to queue", and the test failed on STATE rather
 * than on code. The event was fine; the second run was starting from the first run's leftovers.
 *
 * Filesystem-only, so it is safe regardless of webServer start order — the API reads per-user state
 * from disk per request.
 *
 * Deliberately NOT a fresh random identity per test: a stable id makes the failure legible (you know
 * which flow wrote what), and makes the Umami surface query afterwards attributable. Wiping is the
 * cheaper half of that trade.
 */
export default async function globalSetup(): Promise<void> {
  // Resolved from the Playwright cwd (the config's directory, web/learning-player), NOT from
  // `__dirname` — this file is loaded as ESM, where `__dirname` does not exist. It must match the
  // `APP_DATA_DIR` the config hands the API server, which is written the same relative way.
  const dir = resolve('e2e/.telemetry-state')
  if (existsSync(dir)) {
    rmSync(dir, { recursive: true, force: true })
    // eslint-disable-next-line no-console
    console.log(`[telemetry] wiped ${dir} so this run starts clean`)
  }
  // The reel, "Learn this week" and search read the search index the disposable corpus copy is
  // seeded without: without it their events cannot fire at all (2026-10-11, as for the design tier).
  await buildIndexIfAbsent()
}
