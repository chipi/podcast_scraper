import { expect, test } from '@playwright/test'

/**
 * The legal pages and the way in to them (2026-10-05).
 *
 * Signed OUT on purpose: the sign-in page is where someone decides to create an account, so its
 * footer's Terms of use and Privacy policy links have to open before there is an account at all.
 * The third-party list is the file `scripts/third-party.mjs` generated during THIS build — the
 * preview server serves the real build — so a broken generator fails here, not in a review.
 */
test('the sign-in footer links to the Terms of use and the Privacy policy, signed out', async ({ page }) => {
  await page.goto('/login')
  const legal = page.getByTestId('login-legal')
  await expect(legal).toContainText('By continuing, you agree to our Terms of use and acknowledge our Privacy policy.')

  await page.getByTestId('login-terms').click()
  await expect(page.getByTestId('about-page-title')).toHaveText('Terms of use')
  await expect(page.getByTestId('terms-of-use')).toBeVisible()

  // Back returns to the sign-in page it was opened from, not to Settings.
  await page.getByTestId('about-page-back').click()
  await expect(page.getByTestId('login-legal')).toBeVisible()

  await page.getByTestId('login-privacy').click()
  await expect(page.getByTestId('about-page-title')).toHaveText('Privacy policy')
  await expect(page.getByTestId('privacy-policy')).toBeVisible()
})

test('/terms is a short link to the Terms of use', async ({ page }) => {
  await page.goto('/terms')
  await expect(page.getByTestId('terms-of-use')).toBeVisible()
})

test('third-party software lists the packages this build ships, each with its licence', async ({ page }) => {
  await page.goto('/about/third-party')
  await expect(page.getByTestId('third-party-count')).toContainText('packages')
  // `vue` is unscoped, so it is a row of its own (scoped packages sit inside their @scope group).
  // Top-level rows only: `@sentry/vue` also reads "vue", inside its (closed) @sentry group.
  const vue = page
    .locator('[data-testid="third-party"] > ul > li > [data-testid="third-party-entry"]')
    .filter({ has: page.getByText('vue', { exact: true }) })
  await expect(vue).toBeVisible()
  await vue.locator('summary').click()
  await expect(vue.locator('pre')).toContainText('MIT License')

  // Scoped packages are grouped: @babel opens to its members.
  const babel = page.getByTestId('third-party-group').filter({ hasText: '@babel' })
  await babel.locator('summary').first().click()
  await expect(babel.getByTestId('third-party-entry').first()).toBeVisible()
})
