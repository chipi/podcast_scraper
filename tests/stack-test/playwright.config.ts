import { defineConfig, devices } from '@playwright/test'

const baseURL = process.env.STACK_TEST_BASE_URL ?? 'http://127.0.0.1:8090'
// Tailscale MagicDNS HTTPS on drill/prod often uses certs Playwright does not
// trust out of the box. Set ``STACK_TEST_INSECURE_TLS=1`` from drill GHA only.
const ignoreHTTPSErrors = process.env.STACK_TEST_INSECURE_TLS === '1'

export default defineConfig({
  testDir: '.',
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [['list']],
  timeout: 120_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL,
    ignoreHTTPSErrors,
    trace: process.env.CI ? 'on-first-retry' : 'retain-on-failure',
    ...devices['Desktop Firefox'],
  },
  projects: [{ name: 'firefox', use: {} }],
})
