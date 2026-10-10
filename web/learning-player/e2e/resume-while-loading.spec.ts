import { expect, test } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * Resume on a slow network (Android tester, 2026-10-10): Home's "Resume · 28:17" opened the player
 * at 0:00, scrubber at the start, while the audio then played from 28:17. The element held the
 * resume position the whole time; the time on screen only moved on `timeupdate`, which no element
 * fires before playback starts. The fixture audio is served locally and loads at once, so the
 * window never opened in tests: the audio here is held back for 4 s — delayed, not replaced.
 */
test('Resume shows the saved position while the audio is still loading', async ({ page }, testInfo) => {
  await signInIsolated(page, 'resume-while-loading', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('The Bessent Tape').first().click()
  await expect(page.getByTestId('player-hero')).toBeVisible()
  const slug = decodeURIComponent(page.url().split('/').filter(Boolean).pop()!.split('?')[0])
  // Leave the player BEFORE saving a position: leaving saves the element's own position (0:00 here,
  // nothing played), which would overwrite the one this test sets.
  await page.goto('/podcast/p05')
  await page.waitForTimeout(1000)
  const saved = await page.request.put(`/api/app/playback/${encodeURIComponent(slug)}`, {
    data: { position_seconds: 200, finished: false },
  })
  expect(saved.ok()).toBe(true)

  await page.route('**/audio/**', async (route) => {
    await new Promise((r) => setTimeout(r, 4000))
    await route.continue()
  })
  await page.goto('/')
  for (const name of ['Not now', 'Skip step']) {
    const b = page.getByRole('button', { name, exact: true })
    if (await b.isVisible().catch(() => false)) await b.click().catch(() => {})
  }
  const resume = page.getByTestId('home-resume')
  await expect(resume).toContainText('3:20')
  await resume.click()

  const shown = page.getByTestId('player-times').locator('span').first()
  await expect(shown).toHaveText('3:20', { timeout: 2000 })
  // ...and it is the loading window being tested, not playback that already started.
  const readyState = await page.evaluate(
    () => (document.querySelector('audio[data-testid="app-audio"]') as HTMLAudioElement).readyState,
  )
  expect(readyState, 'the audio should still be loading when the time is read').toBeLessThan(2)
  await expect(page.locator('input[type="range"]').first()).toHaveValue('200')
})
