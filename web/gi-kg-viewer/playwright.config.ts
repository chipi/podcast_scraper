import { defineConfig, devices } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

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
  workers: process.env.CI ? 2 : undefined,
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
  webServer: {
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
    },
    /** Reuse a dev server on 5174 when present so local runs do not fail if `CI=true`. */
    reuseExistingServer: true,
    timeout: process.env.CI ? 180_000 : 120_000,
  },
})
