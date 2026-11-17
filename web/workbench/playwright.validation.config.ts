import { defineConfig, devices } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/**
 * Real-corpus validation config. Drives a real browser against the
 * ``make serve`` dev stack (viewer on 5173, API on 8000) using a
 * real corpus on disk. Walks every matrix surface and takes screenshots
 * for post-hoc inspection. Separate from playwright.config.ts which
 * uses mocks + its own dev server on 5174.
 */
export default defineConfig({
  testDir: './e2e/validation',
  fullyParallel: false, // run sequentially so screenshots are predictable
  workers: 1,
  retries: 0,
  reporter: [['list']],
  timeout: 90_000,
  expect: { timeout: 15_000 },
  outputDir: path.join(__dirname, 'validation-results'),
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'retain-on-failure',
    screenshot: 'on',
    ...devices['Desktop Chrome'],
    viewport: { width: 1440, height: 900 },
  },
  projects: [{ name: 'chromium', use: {} }],
  // No webServer — we expect ``make serve`` already running.
})
