import { rmSync } from 'node:fs'
import { join } from 'node:path'

/**
 * Wipe the per-user e2e state (APP_DATA_DIR = e2e/.app-state) before the run so local runs match
 * CI's clean checkout. ``signInIsolated`` derives a STABLE user id per (test, project), and the
 * state dir is gitignored but persists across local runs — so a prior run's writes (e.g. a Pause
 * click that persists ``resurfacing_settings.paused = true``) leak into the next run and break
 * fresh-user "honest-empty" assertions. Filesystem-only, so it's safe regardless of webServer
 * start order; the API reads per-user state from disk per request.
 */
export default function globalSetup(): void {
  // Playwright runs the config from web/learning-player/, matching the webServer cwd that
  // resolves APP_DATA_DIR = 'e2e/.app-state'.
  rmSync(join(process.cwd(), 'e2e', '.app-state'), { recursive: true, force: true })
}
