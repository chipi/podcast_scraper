import { expect, test } from '@playwright/test'

/**
 * C2 auth flow (#1081): a signed-in session shows the user + sign-out; signing out flips the
 * header back to the sign-in entry. Backend is stubbed (no real OAuth in e2e — the mock
 * provider covers the real code-flow path in the Python integration tests).
 */
test('signed-in header shows sign-out and signing out reverts to sign-in', async ({ page }) => {
  let authed = true
  await page.route('**/api/app/me', (route) =>
    authed
      ? route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ user_id: 'u_1', email: 'dev@localhost', name: 'Dev' }),
        })
      : route.fulfill({ status: 401, contentType: 'application/json', body: '{"detail":"no"}' }),
  )
  await page.route('**/api/app/auth/logout', (route) => {
    authed = false
    return route.fulfill({ status: 204, body: '' })
  })
  await page.route('**/api/app/episodes*', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ items: [], page: 1, page_size: 20, total: 0, has_more: false }),
    }),
  )

  await page.goto('/')
  const signOut = page.getByRole('button', { name: 'Sign out' })
  await expect(signOut).toBeVisible()

  await signOut.click()
  await expect(page.getByRole('link', { name: 'Sign in' })).toBeVisible()
})
