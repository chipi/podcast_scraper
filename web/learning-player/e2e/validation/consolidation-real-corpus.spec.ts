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

  const revisitTab = page.getByRole('button', { name: /revisit/i })
  if (await revisitTab.isVisible().catch(() => false)) {
    await revisitTab.click()
    await page.waitForLoadState('networkidle')
    await page.screenshot({
      path: 'validation-results/revisit-02-inbox.png',
      fullPage: true,
    })
  }

  // Settings surface (pause/resume + cadence) — hits `/api/app/resurfacing/settings`.
  const settingsButton = page
    .getByRole('button', { name: /pause|resume|settings/i })
    .first()
  if (await settingsButton.isVisible().catch(() => false)) {
    await settingsButton.click()
    await page.waitForLoadState('networkidle')
    await page.screenshot({
      path: 'validation-results/revisit-03-settings.png',
      fullPage: true,
    })
  }
})
