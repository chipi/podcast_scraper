import { defineConfig, devices } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/**
 * Browser E2E against Vite dev server (no Python API required for offline tests).
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: [
    ['list'],
    ['html', { open: 'never', outputFolder: 'playwright-report' }],
    ['json', { outputFile: path.join(__dirname, 'e2e-results.json') }],
  ],
  timeout: 60_000,
  expect: { timeout: 15_000 },
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
    /** Reuse a dev server on 5174 when present so local runs do not fail if `CI=true`. */
    reuseExistingServer: true,
    timeout: process.env.CI ? 180_000 : 120_000,
  },
})
