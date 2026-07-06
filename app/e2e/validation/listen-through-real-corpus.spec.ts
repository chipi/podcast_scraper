import { expect, test } from '@playwright/test'

/**
 * Tier-3 — full listen-through against a real backend + real corpus.
 *
 * Regular app-e2e (`app/e2e/full-listen.spec.ts`) exercises the same
 * chain against the committed synthetic corpus. Tier-3 is the DRIFT
 * gate: same walk, real corpus, screenshotted per step, sequential.
 *
 * Assumes `make serve-for-validation` is already up (see
 * `app/e2e/validation/README.md`). Operator-driven corpus via
 * `APP_CORPUS_PATH`; nightly CI defaults to the synthetic fixture.
 */

test('operator listen-through: browse → play → capture → verify', async ({ page }) => {
  // === Home ================================================================
  await page.goto('/')
  await page.waitForLoadState('networkidle')
  await expect(page.getByText('Learning Player')).toBeVisible()
  await page.screenshot({ path: 'validation-results/01-home.png', fullPage: true })

  // === Sign in via the mock provider (make serve-for-validation sets
  //     APP_OAUTH_PROVIDER=mock so the flow is one HTTP round trip) ==========
  await page.goto('/api/app/auth/login?as=tier3-app')
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()
  await page.screenshot({ path: 'validation-results/02-signed-in.png', fullPage: true })

  // === Browse to the first episode from Home ===============================
  await page.goto('/')
  // Real corpus: pick whatever the first "What's new" card exposes.
  const firstEpisodeCard = page
    .locator('[data-testid="episode-card"], article a, [role="link"]')
    .filter({ hasText: /./ })
    .first()
  await firstEpisodeCard.click()
  await page.waitForLoadState('networkidle')
  // Player surface is up (title + a transcript segment or the pending state).
  await expect(
    page.getByRole('button', { name: 'Play', exact: true }),
  ).toBeVisible({ timeout: 20_000 })
  await page.screenshot({ path: 'validation-results/03-player.png', fullPage: true })

  // === Play + advance playhead =============================================
  await page.getByRole('button', { name: 'Play', exact: true }).click()
  await expect(
    page.getByRole('button', { name: 'Pause', exact: true }),
  ).toBeVisible()

  await page.evaluate(() => {
    const audio = document.querySelector('audio')
    if (!audio) return
    audio.currentTime = 60
    audio.dispatchEvent(new Event('timeupdate'))
  })
  await page.screenshot({ path: 'validation-results/04-playing.png', fullPage: true })

  // === Capture a moment ====================================================
  const markMoment = page.getByRole('button', { name: 'Mark this moment' })
  if (await markMoment.isVisible().catch(() => false)) {
    await markMoment.click()
    await page.screenshot({ path: 'validation-results/05-captured.png', fullPage: true })
  }

  // === Verify in Library → Highlights ======================================
  await page.goto('/library')
  await page.waitForLoadState('networkidle')
  await page.screenshot({ path: 'validation-results/06-library.png', fullPage: true })

  const highlightsTab = page.getByRole('button', { name: 'Highlights' })
  if (await highlightsTab.isVisible().catch(() => false)) {
    await highlightsTab.click()
    await page.waitForLoadState('networkidle')
    await page.screenshot({
      path: 'validation-results/07-highlights.png',
      fullPage: true,
    })
  }
})
