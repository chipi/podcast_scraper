import { expect, test } from '@playwright/test'
import { routeLoadableAudio } from './helpers'

/**
 * The transcript is opt-in on mobile: the controls-panel "Transcript" toggle opens AND closes it.
 * On desktop the transcript is the always-visible side column and the toggle is hidden. Runs under
 * both Playwright projects (mobile-chrome + desktop-chrome), asserting the right behaviour for each.
 *
 * routeLoadableAudio: headless can't decode the fixture audio, so the transport panel (which hosts
 * the toggle) would otherwise fall back to the audio-error message. A playable WAV renders the real
 * panel — matching a real device where the bridged audio loads.
 */
test('transcript toggle opens and closes on mobile; side column on desktop', async ({ page }) => {
  await routeLoadableAudio(page)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await page.getByRole('heading', { name: /Index Investing Without the Myths/ }).waitFor()

  const toggle = page.getByTestId('transcript-toggle')
  await toggle.waitFor({ state: 'attached', timeout: 15_000 })
  const line = page.getByText(/Index funds are not a strategy/).first()

  if (await toggle.isVisible()) {
    // Mobile: starts closed → open → closed, with aria-expanded tracking state.
    await expect(line).toBeHidden()
    await expect(toggle).toHaveAttribute('aria-expanded', 'false')

    await toggle.click()
    await expect(line).toBeVisible()
    await expect(toggle).toHaveAttribute('aria-expanded', 'true')

    await toggle.click()
    await expect(line).toBeHidden()
    await expect(toggle).toHaveAttribute('aria-expanded', 'false')
  } else {
    // Desktop: transcript is the always-visible side column (no toggle).
    await expect(line).toBeVisible()
  }
})
