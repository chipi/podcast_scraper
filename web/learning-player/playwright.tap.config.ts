import { defineConfig, devices } from '@playwright/test'
export default defineConfig({
  testDir: './e2e',
  testMatch: 'design-invariants.spec.ts',
  timeout: 240_000,
  expect: { timeout: 10_000 },
  workers: 1,
  reporter: [['list']],
  use: { baseURL: 'http://localhost:4174', trace: 'off' },
  projects: [{ name: 'pixel7', use: { ...devices['Pixel 7'] } }],
})
