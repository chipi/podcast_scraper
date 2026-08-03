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
export default function globalSetup(): void {
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
