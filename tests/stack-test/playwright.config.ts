import { defineConfig, devices } from '@playwright/test'

const baseURL = process.env.STACK_TEST_BASE_URL ?? 'http://127.0.0.1:8090'

export default defineConfig({
  testDir: '.',
  // ``stack-jobs-flow.spec.ts`` is the full UI flow (add feed via UI →
  // select profile → run pipeline job via /api/jobs → validate
  // Library/Digest/Search/Graph after success). It needs a seeded
  // ``feeds.spec.yaml`` at the corpus root and a mock-feeds host that
  // serves the multi-RSS layout the spec references — both pending
  // (TODO: wire seed + multi-RSS into the compose overlay, then drop
  // this ignore). Until then ``stack-viewer.spec.ts`` provides the
  // smoke coverage.
  testIgnore: ['**/stack-jobs-flow.spec.ts'],
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [['list']],
  timeout: 120_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL,
    trace: process.env.CI ? 'on-first-retry' : 'retain-on-failure',
    ...devices['Desktop Firefox'],
  },
  projects: [{ name: 'firefox', use: {} }],
})
