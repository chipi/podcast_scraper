import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

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
  await expect(page.getByTestId('third-party-count')).toContainText('libraries')
  // Build-only tooling is not listed: the Capacitor CLI syncs the native projects and never ships.
  await expect(page.getByTestId('third-party')).not.toContainText('@capacitor/cli')
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

  // Native libraries from the committed iOS / Android snapshot, badged with their platform.
  const androidx = page.getByTestId('third-party-group').filter({ hasText: 'androidx.core' })
  await expect(androidx.getByTestId('third-party-platform')).toHaveText('Android')
  const sentryPod = page
    .locator('[data-testid="third-party"] > ul > li > [data-testid="third-party-entry"]')
    .filter({ has: page.getByText('Sentry', { exact: true }) })
  await expect(sentryPod.getByTestId('third-party-platform')).toHaveText('iOS')
})

test('Terms carries its draft notice; opened directly, Back falls back to Settings', async ({ page }, testInfo) => {
  await signInIsolated(page, 'legal-back-fallback', testInfo)
  await page.goto('/about/terms')
  await expect(page.getByTestId('terms-draft-notice')).toBeVisible()
  // No in-app history to return to (a fresh load), so Back goes to Settings, where these live.
  await page.getByTestId('about-page-back').click()
  await expect(page).toHaveURL(/\/settings$/)
})

test('third-party rows: members named without their prefix, a labelled licence box, the font listed', async ({
  page,
}) => {
  await page.goto('/about/third-party')
  const list = page.getByTestId('third-party')
  await expect(page.getByTestId('third-party-count')).toBeVisible()
  // A scope row opens to its members, each named WITHOUT the "@vue/" prefix the group shows.
  const vueGroup = page.getByTestId('third-party-group').filter({ hasText: '@vue' })
  await vueGroup.locator('summary').first().click()
  const member = vueGroup.getByTestId('third-party-entry').first()
  await expect(member).toBeVisible()
  expect(await member.locator('summary').innerText()).not.toContain('@vue/')
  // Opening a row shows the link and a LABELLED "Licence text" box.
  await member.locator('summary').click()
  await expect(member.getByText('Licence text', { exact: true })).toBeVisible()
  // The self-hosted font is listed with its licence.
  await expect(list.getByText('Google Sans (font)')).toBeVisible()
})
