import { expect, test, type Page } from '@playwright/test'

/**
 * Viewer auth gate + role gating (#1128), with the backend mocked via route interception.
 * Each test signs in as a different role and asserts the UI it should and shouldn't see. The
 * real parallel multi-user isolation is proven in the backend suite (test_app_admin_users.py).
 */

type Role = 'admin' | 'creator' | 'listener'

const USER = (role: Role) => ({
  user_id: `u_${role}`,
  email: `${role}@x.io`,
  name: role[0].toUpperCase() + role.slice(1),
  role,
  disabled: false,
})

/** Host-rooted `/api/` only — must NOT match the viewer's own `/src/api/*.ts` module URLs. */
const API_FALLBACK = /^https?:\/\/[^/]+\/api\//

/** Mock the minimal API surface: a generic ok fallback, then a role-specific (or anon) /me. */
async function signInAs(page: Page, role: Role | null): Promise<void> {
  // Fallback first (least specific) — keeps boot calls (health, digest, artifacts) from hanging.
  await page.route(API_FALLBACK, (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }),
  )
  await page.route('**/api/app/me', (route) => {
    if (role === null) return route.fulfill({ status: 401, body: '{}' })
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(USER(role)),
    })
  })
  await page.route('**/api/app/admin/users', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        USER('admin'),
        { ...USER('creator'), provider: 'mock' },
        { ...USER('listener'), provider: 'mock' },
      ]),
    }),
  )
}

test.describe('viewer auth + roles (mocked API)', () => {
  test('anonymous → login landing', async ({ page }) => {
    await signInAs(page, null)
    await page.goto('/')
    await expect(page.getByTestId('login-button')).toBeVisible()
    await expect(page.getByTestId('main-tab-digest')).toHaveCount(0)
  })

  test('listener → no-access screen, no shell', async ({ page }) => {
    await signInAs(page, 'listener')
    await page.goto('/')
    await expect(page.getByTestId('no-access-message')).toBeVisible()
    await expect(page.getByTestId('no-access-signout')).toBeVisible()
    await expect(page.getByTestId('main-tab-digest')).toHaveCount(0)
  })

  test('creator → base shell only (digest/library/graph); no Dashboard/Ops/Admin', async ({
    page,
  }) => {
    await signInAs(page, 'creator')
    await page.goto('/')
    await expect(page.getByTestId('main-tab-digest')).toBeVisible()
    await expect(page.getByTestId('main-tab-library')).toBeVisible()
    await expect(page.getByTestId('main-tab-graph')).toBeVisible()
    await expect(page.getByTestId('main-tab-dashboard')).toHaveCount(0)
    await expect(page.getByTestId('main-tab-ops')).toHaveCount(0)
    await expect(page.getByTestId('main-tab-admin')).toHaveCount(0)
    // user menu shows the role
    await page.getByTestId('user-menu-button').click()
    await expect(page.getByTestId('user-menu-role')).toHaveText('Creator')
  })

  test('admin → Dashboard + Ops + Admin tabs; Admin opens the user table', async ({ page }) => {
    await signInAs(page, 'admin')
    await page.goto('/')
    await expect(page.getByTestId('main-tab-dashboard')).toBeVisible()
    await expect(page.getByTestId('main-tab-ops')).toBeVisible()
    const adminTab = page.getByTestId('main-tab-admin')
    await expect(adminTab).toBeVisible()
    await adminTab.click()
    await expect(page.getByTestId('users-admin')).toBeVisible()
    await expect(page.getByTestId('user-row-creator@x.io')).toBeVisible()
    // the admin's own row has its role control disabled (self-lockout)
    await expect(page.getByTestId('role-select-admin@x.io')).toBeDisabled()
  })
})
