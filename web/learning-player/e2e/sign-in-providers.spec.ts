import { expect, test, type Page } from '@playwright/test'

/**
 * The real-provider sign-in buttons (Google #2275 naming, Sign in with Apple #2275).
 *
 * The e2e server runs the MOCK provider, whose dev picker replaces these buttons, so two responses
 * are shaped here — and only these two: `/auth/dev-users` says the picker is off (what a prod
 * server says), and `/health` lists the providers a deployment has configured. Everything else is
 * the real server.
 */
async function realProviders(page: Page, providers: string[]): Promise<void> {
  await page.route('**/auth/dev-users', (r) => r.fulfill({ json: { enabled: false, users: [] } }))
  await page.route('**/api/health', async (r) => {
    const resp = await r.fetch()
    await r.fulfill({ response: resp, json: { ...(await resp.json()), auth_providers: providers } })
  })
}

// The app re-checks /health in the background; release the intercepts so one still in flight when a
// test ends is dropped instead of failing the run.
test.afterEach(async ({ page }) => {
  await page.unrouteAll({ behavior: 'ignoreErrors' })
})

test('Google only: the button says it is Google, and there is no Apple button', async ({ page }) => {
  await realProviders(page, ['google'])
  await page.goto('/login')
  await expect(page.getByTestId('signin-button')).toHaveText('Sign in with Google')
  await expect(page.getByTestId('signin-apple-button')).toHaveCount(0)
})

test('Apple configured: both buttons, the same height, each with its own framing', async ({ page }) => {
  await realProviders(page, ['google', 'apple'])
  await page.goto('/login?mode=signup')
  const google = page.getByTestId('signin-button')
  const apple = page.getByTestId('signin-apple-button')
  await expect(google).toHaveText('Sign up with Google')
  await expect(apple).toHaveText('Sign up with Apple')
  // Apple: no smaller than other sign-in buttons. Google: no less prominent than other providers.
  const [g, a] = await Promise.all([google.boundingBox(), apple.boundingBox()])
  expect(g && a && Math.abs(g.height - a.height) < 1).toBe(true)
  expect(g && a && Math.abs(g.width - a.width) < 1).toBe(true)
})

test('the Apple button starts the Apple flow, not the primary provider', async ({ page }) => {
  await realProviders(page, ['google', 'apple'])
  await page.goto('/login')
  const started = page.waitForRequest((req) => req.url().includes('/api/app/auth/login'))
  await page.getByTestId('signin-apple-button').click()
  expect(new URL((await started).url()).searchParams.get('provider')).toBe('apple')
})
