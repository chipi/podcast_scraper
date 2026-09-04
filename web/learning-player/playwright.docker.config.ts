/**
 * Runs the full e2e suite against a CONTAINERISED api instead of `.venv/bin/python serve`.
 *
 * Use `make test-app-e2e-docker`, which builds the image if needed, seeds the fixture corpus
 * into a volume, waits for health, runs this config and reaps the container even on failure.
 * Running `npx playwright test --config=…` directly works too, but then the api on :8011 is
 * yours to start and stop.
 *
 * Why it exists: `make test-app-e2e` boots the api from the local venv, which needs the
 * `[search]` extra — and on x86-64 macOS that extra CANNOT install, because `torch>=2.11` and
 * `lancedb>=0.33` publish no Intel-Mac wheels. The venv api then answers every search with
 * `no_index` and the grounded-search specs fail for reasons unrelated to the code. The api image
 * carries the pinned search stack and bakes the MiniLM embedding model at
 * `HF_HOME=/opt/podcast_hf`, so the same suite passes there.
 *
 * Differences from `playwright.config.ts`: no api `webServer` (the container is the api), and
 * `--workers=4` is expected — full parallelism saturates one uvicorn container and produces
 * load-induced failures that look like product bugs.
 */
import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  testIgnore: ['**/validation/**', '**/live/**'],
  fullyParallel: true,
  globalSetup: './e2e/globalSetup.ts', // wipes e2e/.app-state; skips index build (index present)
  timeout: 60_000,
  expect: { timeout: 15_000 },
  retries: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:4174',
    trace: 'on-first-retry',
    serviceWorkers: 'block',
  },
  projects: [
    { name: 'mobile-chrome', use: { ...devices['Pixel 7'] } },
    { name: 'desktop-chrome', use: { ...devices['Desktop Chrome'] } },
  ],
  // NOTE: no API webServer here — the API runs in Docker on :8011 (started by hand).
  webServer: [
    {
      command: '../../.venv/bin/python ../../scripts/tools/run_e2e_mock_server.py --port 18765',
      url: 'http://127.0.0.1:18765/audio/p05_e03.mp3',
      reuseExistingServer: true,
      timeout: 120_000,
      env: { PYTHONPATH: '../../src:../..' },
    },
    {
      command: 'npm run build && npm run preview -- --port 4174 --strictPort --host 127.0.0.1',
      url: 'http://127.0.0.1:4174',
      reuseExistingServer: true,
      timeout: 180_000,
      env: { VITE_API_TARGET: 'http://127.0.0.1:8011' },
    },
  ],
})
