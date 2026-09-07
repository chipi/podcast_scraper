import { expect, test } from '@playwright/test'

/**
 * Smoke — REAL API over the COMMITTED validation corpus
 * (tests/fixtures/app-validation-corpus/v3, built by scripts/build_app_validation_corpus.py),
 * NO mocks. RFC-120 login-first: a logged-out visitor is redirected to the /welcome lure landing,
 * which shows the value-prop hero, a "Create your free account" CTA, and a "Featured this week"
 * teaser rail sourced from the real /discover endpoint.
 */
test('logged-out visit to / lands on the /welcome lure landing', async ({ page }) => {
  await page.goto('/')
  await expect(page).toHaveURL(/\/welcome/)
  // Value-prop hero (landing.heroTitle i18n key)
  await expect(page.getByText('Understand any podcast in minutes.')).toBeVisible()
  // Primary CTA (landing.ctaCreate)
  await expect(page.getByTestId('landing-cta-primary')).toBeVisible()
  await expect(page.getByTestId('landing-cta-primary')).toHaveText('Create your free account')
  // Sign-in affordance also present
  await expect(page.getByTestId('landing-cta-signin')).toBeVisible()
  // Featured teaser rail sources from the real /discover endpoint
  await expect(page.getByTestId('landing-featured')).toBeVisible()
  await expect(page.getByTestId('landing-card').first()).toBeVisible()
})
