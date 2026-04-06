import { defineConfig, devices } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/**
 * RFC-062 M7: browser E2E against Vite dev server (no Python API required for offline tests).
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }]],
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    /* Dedicated port so local `npm run dev` on 5173 does not collide with E2E. */
    baseURL: 'http://127.0.0.1:5174',
    trace: 'on-first-retry',
    ...devices['Desktop Firefox'],
  },
  projects: [{ name: 'firefox', use: {} }],
  webServer: {
    // npm exec: avoid npx install prompts on CI; --host 127.0.0.1 matches baseURL (IPv4)
    command:
      'npm exec vite -- --port 5174 --strictPort --host 127.0.0.1',
    cwd: __dirname,
    url: 'http://127.0.0.1:5174',
    reuseExistingServer: !process.env.CI,
    timeout: process.env.CI ? 180_000 : 120_000,
  },
})
