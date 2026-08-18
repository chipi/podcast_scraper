import { defineConfig, devices } from '@playwright/test'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '..', '..')
/**
 * The API server below must run the interpreter that HAS podcast_scraper installed. CI creates
 * the venv at <repo>/.venv and never activates it, so a bare `python` there is the setup-python
 * interpreter — which would start, fail to import podcast_scraper, and surface as every spec
 * timing out on ECONNREFUSED rather than as a missing install.
 */
const venvPython = path.join(repoRoot, '.venv', 'bin', 'python')
const pythonBin = existsSync(venvPython) ? venvPython : 'python3'

/**
 * Browser E2E against Vite dev server (no Python API required for offline tests).
 */
export default defineConfig({
  testDir: './e2e',
  // ``e2e/validation/`` holds Tier-3 specs that require a running ``make
  // serve`` stack + an operator-supplied ``CORPUS_PATH`` env var. They
  // hard-error at module load when ``CORPUS_PATH`` is unset (intentional —
  // see ``e2e/validation/real-corpus.spec.ts`` header), so the default
  // viewer-e2e GHA job must not pick them up. They run under their own
  // ``playwright.validation.config.ts`` invoked by ``make ci-ui-validation``.
  // ``e2e/live/`` is the #43 post-deploy smoke vs the LIVE operator.closelistening.app — it
  // runs under ``playwright.live.config.ts`` against the deployed origin, NOT this local stack.
  testIgnore: ['**/validation/**', '**/live/**'],
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  // Local runs get 1 retry too: the production-shaped handoff specs assert on
  // millisecond-scale supersession ordering and flake under laptop worker
  // contention. CI gets 2 retries (3 attempts) so timing-sensitive specs like the
  // 15s stuck-handoff (H6.3) survive transient runner contention; local keeps 1.
  retries: process.env.CI ? 2 : 1,
  /**
   * ONE worker, everywhere. Four operator specs (cron-preview, feed-overrides, scheduled-jobs,
   * status-bar-feeds-operator) write to the SHARED corpus the API serves, so two of them running
   * in different workers overwrite each other's seeded state. Since #1619 gave the suite a real
   * backend that stopped being theoretical: at 2 workers CI failed exactly those four, with the
   * guard they carry — "writes to the shared corpus and cannot run with 2 workers … re-run with
   * --workers=1" — which is also the default `e2e/run-local-stack.sh` has always used.
   *
   * The cost is wall-clock, not coverage: 218 passed in 7.7m at 2 workers, so expect roughly
   * double, comfortably inside this job's 45m budget. Parallelism here would have to be bought by
   * giving each worker its own corpus and API, which is a bigger change than the time is worth.
   */
  workers: 1,
  reporter: [
    ['list'],
    ['html', { open: 'never', outputFolder: 'playwright-report' }],
    ['json', { outputFile: path.join(__dirname, 'e2e-results.json') }],
  ],
  // #1619 — raised from 60s/15s when the suite moved onto a real backend.
  //
  // A mocked `/api/search` returned instantly; a live one runs a query embedding plus a LanceDB
  // search, and the Tier-2 walks issue several per test. On a contended machine (the container
  // runtime alone is a VM at ~190% CPU) that pushed `search-production/rail-launch` and the
  // dashboard Intelligence tab past the old 60s cap — as TIMEOUTS, not wrong assertions, which is
  // the tell that the budget was the problem rather than the specs.
  //
  // These are ceilings, not waits: a healthy run does not get near them.
  timeout: 120_000,
  expect: { timeout: 20_000 },
  use: {
    /* Dedicated port so local `npm run dev` on 5173 does not collide with E2E. */
    baseURL: 'http://127.0.0.1:5174',
    trace: process.env.CI ? 'on-first-retry' : 'retain-on-failure',
    ...devices['Desktop Firefox'],
  },
  projects: [{ name: 'firefox', use: {} }],
  webServer: [
    {
      /**
       * The API the suite talks to since #1619 moved it off route-fulfilled mocks.
       *
       * `reuseExistingServer` is what keeps BOTH workflows working from one config: when the
       * container from `e2e/run-local-stack.sh` (or `make test-ui-e2e-live`) is already serving
       * 8012, Playwright reuses it and never starts this one — that path bakes in its own model
       * cache and stays the one to use on a machine that cannot install `[search]` at all. With
       * nothing on 8012, as in CI, this native server starts instead.
       *
       * The env below is not optional decoration. Without `APP_SIGNUP_MODE=open`,
       * `/api/app/auth/login?as=…` returns 403 and `signInIsolated` cannot make a session;
       * without `APP_ADMIN_EMAILS` an admin sign-in lands in `creator` so admin-only surfaces
       * never render; without the three `PODCAST_SERVE_ENABLE_*` the operator routes are NOT
       * MOUNTED — `/api/feeds` and `/api/operator-config` 404 rather than 403, which reads as a
       * broken frontend rather than a missing flag.
       */
      // `prepare-corpus.mjs` is chained into the command rather than run from `globalSetup`
      // because Playwright starts webServer BEFORE globalSetup — seeding there lost the race and
      // the server exited 2 on a corpus directory that did not exist yet.
      command: `node e2e/prepare-corpus.mjs && ${pythonBin} -m podcast_scraper.cli serve --output-dir .e2e-corpus/v3 --port 8012 --host 127.0.0.1`,
      cwd: __dirname,
      url: 'http://127.0.0.1:8012/api/health',
      env: {
        ...process.env,
        PYTHONPATH: path.join(repoRoot, 'src'),
        APP_OAUTH_PROVIDER: 'mock',
        APP_SESSION_SECRET: 'e2e-secret',
        APP_SIGNUP_MODE: 'open',
        APP_ADMIN_EMAILS: 'ada-admin@e2e.local',
        APP_DATA_DIR: path.join(__dirname, 'e2e', '.app-state'),
        HF_HUB_OFFLINE: '1',
        TRANSFORMERS_OFFLINE: '1',
        PODCAST_SERVE_ENABLE_FEEDS_API: '1',
        PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API: '1',
        PODCAST_SERVE_ENABLE_JOBS_API: '1',
      },
      reuseExistingServer: true,
      timeout: process.env.CI ? 180_000 : 120_000,
    },
    {
    // npm exec: avoid npx install prompts on CI; --host 127.0.0.1 matches baseURL (IPv4)
    // VITE_DEFAULT_GRAPH_LENS_DAYS=0 — test fixtures use static publish_date values
    // ("2026-04-18", "2024-06-05", …) and would fall outside the production 7-day
    // graph lens once the wall clock advances. All-time lens keeps fixtures stable.
    command:
      'VITE_DEFAULT_GRAPH_LENS_DAYS=0 npm exec vite -- --port 5174 --strictPort --host 127.0.0.1',
    cwd: __dirname,
    url: 'http://127.0.0.1:5174',
    env: {
      ...process.env,
      VITE_DEFAULT_GRAPH_LENS_DAYS: '0',
      // E2E dev server must emit no analytics: hard-disable the dev-default Umami
      // (it would otherwise inject the tracking script in `vite dev`).
      VITE_ANALYTICS_OFF: '1',
      // Proxy /api at the server above rather than vite.config.ts's 8000 default.
      VITE_API_TARGET: 'http://127.0.0.1:8012',
    },
      /** Reuse a dev server on 5174 when present so local runs do not fail if `CI=true`. */
      reuseExistingServer: true,
      timeout: process.env.CI ? 180_000 : 120_000,
    },
  ],
})
