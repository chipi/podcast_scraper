import { expect, test } from '@playwright/test'

/**
 * C1 smoke: the app shell boots, renders the masthead, and a signed-out user sees the
 * sign-in entry. The full listen→capture + a11y (axe) path lands with the Player (#1083).
 * The catalog list call is stubbed so the smoke needs no backend.
 */
test('app shell renders with sign-in for a signed-out user', async ({ page }) => {
  // Signed out: /me → 401; empty catalog so the view settles deterministically.
  await page.route('**/api/app/me', (route) =>
    route.fulfill({ status: 401, contentType: 'application/json', body: '{"detail":"no"}' }),
  )
  await page.route('**/api/app/episodes*', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ items: [], page: 1, page_size: 20, total: 0, has_more: false }),
    }),
  )

  await page.goto('/')
  // Masthead brand (text) + the catalog page heading + the signed-out sign-in entry.
  await expect(page.getByText('Learning Player')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Your Library' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Sign in' })).toBeVisible()
})
