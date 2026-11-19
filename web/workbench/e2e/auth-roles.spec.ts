import { expect, test, type Page } from '@playwright/test'

/**
 * Viewer auth gate + role gating (#1128).
 *
 * Each test signs in as a different role and asserts the UI it should and shouldn't see. The real
 * parallel multi-user isolation is proven in the backend suite (test_app_admin_users.py).
 *
 * #1619 — migrated to the live API, and this file gained its point in the process.
 *
 * It used to fabricate `/api/app/auth/status` with a hand-written `{ role }` and assert the shell
 * reacted. That tests `App.vue`'s `v-if`s against a payload the test wrote — the *server's* role
 * decision, which is what #1128 is about, was never exercised. The mock could have said `admin`
 * for a user the backend considers a listener and the test would still pass.
 *
 * Now the server decides. Verified against the live backend:
 *
 * | sign-in                                   | role the server assigns |
 * | ----------------------------------------- | ----------------------- |
 * | `?as=<new-id>` (no grant)                 | `listener`              |
 * | `?as=<new-id>&grant=creator`              | `creator`               |
 * | `?as=ada-admin` (in `APP_ADMIN_EMAILS`)   | `admin`                 |
 *
 * Requires the server started with `APP_OAUTH_PROVIDER=mock`, `APP_SIGNUP_MODE=open` and
 * `APP_ADMIN_EMAILS=ada-admin@e2e.local` — see e2e/README.md.
 */

/** Server-side sign-in for a fresh identity; returns the email the mock provider synthesizes. */
async function signInAsRole(
  page: Page,
  id: string,
  opts: { grant?: 'creator' } = {},
): Promise<string> {
  const grant = opts.grant ? `&grant=${opts.grant}` : ''
  await page.goto(`/api/app/auth/login?as=${encodeURIComponent(id)}${grant}`)
  return `${id}@e2e.local`
}

test.describe('viewer auth + roles', () => {
  test('anonymous → login landing', async ({ page }) => {
    // No sign-in at all: the real gate sees no session.
    await page.goto('/')

    /* `LoginView` branches on the OAuth provider: `login-button` is the REAL-provider control,
     * while the mock provider renders the dev sign-in form. The e2e stack runs
     * `APP_OAUTH_PROVIDER=mock`, so `login-button` is the wrong branch here — the old mocked
     * version asserted it and passed only because the catch-all `{}` left the app unable to see
     * that a dev provider was configured. Migrating surfaced that the test had been checking a
     * branch this environment never renders. */
    await expect(page.getByTestId('dev-custom-input')).toBeVisible()
    await expect(page.getByTestId('dev-custom-submit')).toBeVisible()
    await expect(page.getByTestId('login-button')).toHaveCount(0)

    // Either way, the gate holds: no shell for an anonymous visitor.
    await expect(page.getByTestId('main-tab-digest')).toHaveCount(0)
  })

  test('listener → no-access screen, no shell', async ({ page }, testInfo) => {
    // A brand-new identity with no grant: the SERVER decides this is a listener.
    await signInAsRole(page, `auth-listener-${testInfo.project.name}`)
    await page.goto('/')
    await expect(page.getByTestId('no-access-message')).toBeVisible()
    await expect(page.getByTestId('no-access-signout')).toBeVisible()
    await expect(page.getByTestId('main-tab-digest')).toHaveCount(0)
  })

  test('creator → base shell only (digest/library/graph); no Dashboard/Ops/Admin', async ({
    page,
  }, testInfo) => {
    await signInAsRole(page, `auth-creator-${testInfo.project.name}`, { grant: 'creator' })
    await page.goto('/')
    await expect(page.getByTestId('main-tab-digest')).toBeVisible()
    await expect(page.getByTestId('main-tab-library')).toBeVisible()
    await expect(page.getByTestId('main-tab-graph')).toBeVisible()
    await expect(page.getByTestId('main-tab-dashboard')).toHaveCount(0)
    await expect(page.getByTestId('main-tab-ops')).toHaveCount(0)
    await expect(page.getByTestId('main-tab-admin')).toHaveCount(0)
    // user menu shows the role the server assigned
    await page.getByTestId('user-menu-button').click()
    await expect(page.getByTestId('user-menu-role')).toHaveText('Creator')
  })

  test('admin → Dashboard + Ops + Admin tabs; Admin opens the user table', async ({ page }) => {
    /* Seed a creator first, in this same browser context, so the admin table has a known
     * non-admin row to assert on. The table is the real user store, which accumulates identities
     * across runs — so assert specific rows rather than counts. */
    const creatorEmail = await signInAsRole(page, 'auth-admin-fixture-creator', {
      grant: 'creator',
    })

    await page.goto('/api/app/auth/login?as=ada-admin')
    await page.goto('/')
    await expect(page.getByTestId('main-tab-dashboard')).toBeVisible()
    await expect(page.getByTestId('main-tab-ops')).toBeVisible()
    const adminTab = page.getByTestId('main-tab-admin')
    await expect(adminTab).toBeVisible()
    await adminTab.click()
    await expect(page.getByTestId('users-admin')).toBeVisible()
    await expect(page.getByTestId(`user-row-${creatorEmail}`)).toBeVisible()
    // the admin's own row has its role control disabled (self-lockout)
    await expect(page.getByTestId('role-select-ada-admin@e2e.local')).toBeDisabled()
  })
})
