import { expect, test } from '@playwright/test'

/**
 * Tier-3 — Library Revisit resurfacing ladder against a real backend.
 *
 * PRD-041 §Resurfacing spec: past highlights resurface on a spaced
 * ladder (2d/1w/1mo/3mo). Signed-out or fresh account has an empty
 * inbox — this walk confirms the surface renders that state honestly
 * (rather than crashing on empty resurfacing.json).
 *
 * Real regression here would be a shape drift in `/api/app/resurfacing`
 * that the fixture-based fast-e2e mocks miss.
 */

test('operator revisit inbox: empty state + settings surface', async ({ page }) => {
  await page.goto('/api/app/auth/login?as=tier3-app-revisit')
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()

  // Library Revisit tab — first-time users see the empty state.
  await page.goto('/library')
  await page.waitForLoadState('networkidle')
  await page.screenshot({
    path: 'validation-results/revisit-01-library.png',
    fullPage: true,
  })

  // Everything below used to be wrapped in `if (visible)`, so this walk contained NO assertion
  // at all beyond the sign-in — it navigated, screenshotted, and could not fail for any reason.
  // Library's tabs are rendered unconditionally, so they are asserted.
  for (const tab of ['Following', 'Saved', 'Collections', 'Revisit']) {
    await expect(page.getByRole('button', { name: tab, exact: true })).toBeVisible()
  }

  await page.getByRole('button', { name: 'Revisit', exact: true }).click()
  await page.waitForLoadState('networkidle')
  await page.screenshot({
    path: 'validation-results/revisit-02-inbox.png',
    fullPage: true,
  })

  // The empty state is the assertion for a fresh account: the inbox must SAY it is empty rather
  // than render a blank panel, which is the regression an operator would otherwise have to spot
  // by eye in a screenshot.
  await expect(page.getByText(/nothing to revisit right now/i)).toBeVisible()

  // Settings surface (pause/resume + cadence) — hits `/api/app/resurfacing/settings`.
  const pause = page.getByRole('button', { name: /^(pause|resume)$/i }).first()
  await expect(pause).toBeVisible()
  await pause.click()
  await page.waitForLoadState('networkidle')
  await page.screenshot({
    path: 'validation-results/revisit-03-settings.png',
    fullPage: true,
  })
})
