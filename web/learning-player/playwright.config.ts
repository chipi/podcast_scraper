import { defineConfig, devices } from '@playwright/test'

/**
 * E2E config for the consumer Learning Player. Boots the Vite preview server and runs the
 * smoke + (later) full listen→capture specs. Mobile-first: the default project emulates a
 * phone viewport, matching the app's primary target (UXS-011).
 */
export default defineConfig({
  testDir: './e2e',
  // e2e/validation/** is Tier-3 — runs under playwright.validation.config.ts
  // against a separately-booted `make serve-for-validation` stack, NOT the
  // fast-tier preview here. Excluding it prevents these specs from firing
  // twice + failing because the shared serviceWorkers:'block' default is
  // wrong for the SW-driven validation walks.
  // e2e/live/** is the #43 post-deploy smoke vs the LIVE closelistening.app — it runs under
  // playwright.live.config.ts against the deployed origin, NOT this local preview stack.
  testIgnore: ['**/validation/**', '**/live/**'],
  fullyParallel: true,
  // The heavy auth-gated specs (capture, consolidation) sign in as ISOLATED per-(spec,project) mock
  // identities (see e2e/helpers.ts) so they never share per-user files WITHIN a run. But those ids
  // are STABLE across runs and APP_DATA_DIR (e2e/.app-state) is gitignored, not cleaned — so a prior
  // run's writes (e.g. a Pause click persisting resurfacing_settings.paused=true) leaked into the
  // next local run and broke fresh-user "honest-empty" assertions. globalSetup wipes that dir so
  // every local run starts clean, matching CI's fresh checkout.
  globalSetup: './e2e/globalSetup.ts',
  // Parity with the stable gi-kg-viewer config (225 specs, green under the same full-`make ci`
  // load): the emulated mobile-chrome (Pixel 7) render + real-API round-trip can exceed the
  // default 5 s expect deadline when the machine is saturated at the end of a full-ci run — which
  // is exactly when `make ci` invokes this stage LOCALLY (process.env.CI unset → previously 0
  // retries + a 5 s deadline). Give the assertions headroom (15 s) and one local retry so a
  // load-induced slow first paint self-heals instead of failing the gate. This matches the
  // deadline to the real render time; it does not paper over a product bug (isolated, these specs
  // pass in 0.5–5 s).
  timeout: 60_000,
  expect: { timeout: 15_000 },
  retries: process.env.CI ? 2 : 1,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: 'http://127.0.0.1:4174',
    trace: 'on-first-retry',
    // Block the PWA service worker so the e2e exercises the real network path deterministically
    // (the SW would otherwise intercept /api/app/* with stale-while-revalidate).
    serviceWorkers: 'block',
  },
  projects: [
    { name: 'mobile-chrome', use: { ...devices['Pixel 7'] } },
    { name: 'desktop-chrome', use: { ...devices['Desktop Chrome'] } },
  ],
  // Full-stack, NO MOCKS: the real consumer API serves a COMMITTED, deterministically-synthesized
  // corpus, and the built app is proxied to it (same-origin via preview proxy). This catches
  // server-contract bugs (e.g. the transcript_file_path metadata key) that a client-mocked e2e
  // cannot.
  //
  // The corpus is tests/fixtures/app-validation-corpus/v3 — checked in, version-pinned, and built
  // by scripts/build_app_validation_corpus.py (deterministic, no pipeline, no ML). There is NO
  // build step here: `serve` reads the committed corpus directly, so boot is fast and stable.
  // Per-user runtime state (queue/profile/interests the API writes) is redirected via APP_DATA_DIR
  // to a gitignored ephemeral dir so the committed corpus tree is never mutated.
  webServer: [
    {
      // Mock podcast host (#1618) — serves the REAL fixture audio the corpus points at.
      //
      // `content.media_url` is a relative `/audio/<episode_id>.mp3`, matching the RSS fixtures'
      // convention, and vite's `/audio` proxy forwards it here. Before this, the corpus carried an
      // undecodable data URI and every transport spec route-mocked a synthetic WAV — so the suite
      // was testing the stub rather than the player.
      //
      // Same server the pytest E2E suite uses; it resolves `tests/fixtures/audio/<FIXTURES_VERSION>`
      // itself, so the version bump is not duplicated here. `reuseExistingServer` means a locally
      // running mock host (or the nginx `mock-feeds` container, for machines whose venv cannot run
      // this) is reused instead of a second bind.
      command: '../../.venv/bin/python ../../scripts/tools/run_e2e_mock_server.py --port 18765',
      url: 'http://127.0.0.1:18765/audio/p05_e03.mp3',
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
      env: { PYTHONPATH: '../../src:../..' },
    },
    {
      // Paths are relative to this config's cwd — web/learning-player/ —
      // so `../..` traverses back to the repo root (where .venv, src/,
      // and tests/ live). Missed this on slice 14; caught by
      // ``make test-app-e2e`` locally when the first run of playwright
      // failed with `../.venv/bin/python: No such file or directory`.
      command:
        '../../.venv/bin/python -m podcast_scraper.cli serve ' +
        '--output-dir ../../tests/fixtures/app-validation-corpus/v3 --port 8011 --host 127.0.0.1',
      url: 'http://127.0.0.1:8011/api/health',
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
      env: {
        PYTHONPATH: '../../src',
        // Use the cached MiniLM embedding model offline — the serve embeds the search query at
        // request time, and without this it tries to reach huggingface.co, fails, and every
        // /api/*/search returns embed_failed ("Search needs the library index"). The model is
        // present locally (dev venv) and preloaded in CI (python-app.yml app-e2e). Pairs with the
        // two-tier index e2e/globalSetup.ts builds, so search returns real grounded results.
        HF_HUB_OFFLINE: '1',
        TRANSFORMERS_OFFLINE: '1',
        // The offline flags alone aren't enough — the model-load path needs the cache dir set
        // explicitly (an inherited HOME is not sufficient). Default to the standard local location;
        // CI sets HF_HOME to the runner cache where preload_ml_models.py stored MiniLM.
        HF_HOME: process.env.HF_HOME || `${process.env.HOME}/.cache/huggingface`,
        HF_HUB_CACHE: process.env.HF_HUB_CACHE || `${process.env.HOME}/.cache/huggingface/hub`,
        APP_OAUTH_PROVIDER: 'mock',
        APP_SESSION_SECRET: 'e2e-secret',
        // Allow the mock dev identity through the access policy (default is allowlist/deny).
        APP_SIGNUP_MODE: 'open',
        // Personalized discovery ON so the recommender A/B (recommendation.spec) can assert the feed
        // re-ranks toward a followed interest. With no interests the feed is recency (unchanged), so
        // this is inert for every other spec.
        APP_PERSONALIZED_RANKING: 'true',
        // RFC-103: pin the momentum "now" so trending is deterministic (the corpus's newest episode
        // is 2026-07-16 → anchor just after it, when the risk/systems content is freshest → rising).
        APP_TRENDING_NOW: '2026-07-20T00:00:00Z',
        // Keep per-user writes (queue/profile/interests) OUT of the committed corpus tree.
        // Relative to the webServer cwd (web/learning-player/); the server resolve()s it against cwd.
        APP_DATA_DIR: 'e2e/.app-state',
      },
    },
    {
      command: 'npm run build && npm run preview -- --port 4174 --strictPort --host 127.0.0.1',
      url: 'http://127.0.0.1:4174',
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      env: { VITE_API_TARGET: 'http://127.0.0.1:8011' },
    },
  ],
})
