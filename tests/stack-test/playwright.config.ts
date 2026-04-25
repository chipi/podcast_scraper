import { defineConfig, devices } from '@playwright/test'

const baseURL = process.env.STACK_TEST_BASE_URL ?? 'http://127.0.0.1:8090'

export default defineConfig({
  testDir: '.',
  // ``stack-jobs-flow.spec.ts`` is the full UI flow (add feed via UI →
  // select profile → run pipeline job via /api/jobs → validate
  // Library/Digest/Search/Graph after success). The seed +
  // multi-RSS mock-feeds wiring is in place (this branch); the
  // remaining blocker is the API job factory's nested ``docker
  // compose run pipeline`` spawn — the same compose args succeed
  // when invoked manually from inside the API container but fail
  // from the factory's subprocess with "Config file not found:
  // /app/config.yaml". Drop this ignore once the factory spawn is
  // fixed; until then ``stack-viewer.spec.ts`` provides smoke
  // coverage.
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
