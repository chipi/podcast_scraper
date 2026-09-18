import { execFileSync } from 'node:child_process'
import { existsSync, readdirSync, rmSync } from 'node:fs'
import { join, resolve } from 'node:path'

/**
 * Wipe the per-user e2e state (APP_DATA_DIR = e2e/.app-state) before the run so local runs match
 * CI's clean checkout. ``signInIsolated`` derives a STABLE user id per (test, project), and the
 * state dir is gitignored but persists across local runs — so a prior run's writes (e.g. a Pause
 * click that persists ``resurfacing_settings.paused = true``) leak into the next run and break
 * fresh-user "honest-empty" assertions. Filesystem-only, so it's safe regardless of webServer
 * start order; the API reads per-user state from disk per request.
 *
 * Then build the two-tier search index if it is absent. The corpus (app-validation-corpus/v3) is
 * committed but its LanceDB ``search/lance_index/`` is gitignored (binary + lance-format-version
 * coupling), and several routes the topic page hits branch on ``has_index`` — so an absent index
 * silently changes which grounded claim a perspective surfaces, and index-dependent specs
 * (perspectives, search) fail. Local runs previously "passed" only when a stale index happened to
 * be lying around, and CI (which never built one) failed. Building it here makes both
 * deterministic. Requires the ``[search]`` extras + the cached MiniLM model (offline); CI provides
 * both before invoking Playwright (see .github/workflows/python-app.yml app-e2e).
 */
/**
 * Refuse to run against an already-listening API that cannot embed a search query.
 *
 * `reuseExistingServer: !CI` means a server left over from an earlier session is adopted silently.
 * If that one was started without `HF_HUB_CACHE`/`HF_HUB_OFFLINE` — trivially easy, since starting
 * the API by hand does not set them — every `/api/search` returns `embed_failed`, and six specs
 * fail with "element(s) not found" for a results heading. Nothing in that output names the cause,
 * and the server is invisible: the run looks like a code regression in search (operator 2026-09-18,
 * where it cost a full diagnosis to find a server started 30 minutes earlier).
 *
 * `/api/health` cannot catch this — a misconfigured server is perfectly healthy. So probe the thing
 * the suite actually depends on, and only when something is ALREADY listening, which is the sole
 * case where reuse can happen. If the port is free, Playwright starts a correctly-configured server
 * and there is nothing to check.
 */
async function rejectMisconfiguredReusedApi(): Promise<void> {
  const base = 'http://127.0.0.1:8011'
  let probe: Response
  try {
    probe = await fetch(`${base}/api/search?q=investing`, {
      signal: AbortSignal.timeout(5000),
    })
  } catch {
    return // nothing listening — Playwright will start its own, correctly configured
  }

  const body = (await probe.json().catch(() => ({}))) as { error?: string; detail?: string }
  if (!body.error) return

  throw new Error(
    `An API is already listening on ${base} and cannot serve search: ${body.error}\n` +
      `  ${(body.detail ?? '').split('\n')[0]}\n\n` +
      `Playwright reuses it (reuseExistingServer), so the search specs would fail with a missing\n` +
      `results heading and no hint as to why. It was almost certainly started by hand, or by an\n` +
      `earlier run, without the HF cache env the config sets.\n\n` +
      `Kill it and re-run:  lsof -ti :8011 | xargs kill\n` +
      `If the model itself is missing:  make preload-ml-models`,
  )
}

export default async function globalSetup(): Promise<void> {
  await rejectMisconfiguredReusedApi()

  // Playwright runs the config from web/learning-player/, matching the webServer cwd that
  // resolves APP_DATA_DIR = 'e2e/.app-state'.
  rmSync(join(process.cwd(), 'e2e', '.app-state'), { recursive: true, force: true })

  const repoRoot = resolve(process.cwd(), '..', '..')
  const corpus = join(repoRoot, 'tests', 'fixtures', 'app-validation-corpus', 'v3')
  const lanceDir = join(corpus, 'search', 'lance_index')
  const hasIndex = existsSync(lanceDir) && readdirSync(lanceDir).length > 0
  if (hasIndex) return

  const python = join(repoRoot, '.venv', 'bin', 'python')
  // eslint-disable-next-line no-console
  console.log('[globalSetup] building two-tier search index for app-validation-corpus/v3…')
  execFileSync(python, ['-m', 'podcast_scraper.cli', 'index-two-tier', '--output-dir', corpus], {
    cwd: repoRoot,
    stdio: 'inherit',
    env: {
      ...process.env,
      PYTHONPATH: join(repoRoot, 'src'),
      // Use the cached embedding model — index_corpus runs allow_download=False, so the model
      // must already be in the HF cache (CI preloads it; the local dev venv has it).
      HF_HUB_OFFLINE: '1',
      TRANSFORMERS_OFFLINE: '1',
    },
  })
}
