import { expect, test } from '@playwright/test'

/**
 * Tier-3 — full listen-through against a real backend + real corpus.
 *
 * Regular app-e2e (`web/learning-player/e2e/full-listen.spec.ts`) exercises the same
 * chain against the committed synthetic corpus. Tier-3 is the DRIFT
 * gate: same walk, real corpus, screenshotted per step, sequential.
 *
 * Assumes `make serve-for-validation` is already up (see
 * `web/learning-player/e2e/validation/README.md`). Operator-driven corpus via
 * `APP_CORPUS_PATH`; nightly CI defaults to the synthetic fixture.
 */

test('operator listen-through: browse → play → capture → verify', async ({ page }) => {
  // === Home ================================================================
  await page.goto('/')
  await page.waitForLoadState('networkidle')
  // The app SHELL rendered. Asserted via the brand wordmark in the banner rather than
  // "Learning Player", which exists only as the document <title> — `getByText` matches visible
  // text, never a <title>, so that assertion could not pass whatever the app did. The primary
  // nav is deliberately NOT used: it is the mobile tab bar, and Tier-3 runs a desktop viewport.
  await expect(page.getByRole('link', { name: /Close Listening/i }).first()).toBeVisible()
  await page.screenshot({ path: 'validation-results/01-home.png', fullPage: true })

  // === Sign in via the mock provider (make serve-for-validation sets
  //     APP_OAUTH_PROVIDER=mock so the flow is one HTTP round trip) ==========
  await page.goto('/api/app/auth/login?as=tier3-app')
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()
  await page.screenshot({ path: 'validation-results/02-signed-in.png', fullPage: true })

  // === Browse to the first episode from Home ===============================
  await page.goto('/')
  // Real corpus: take the first thing on Home that actually navigates to an episode.
  //
  // The previous selector (`[data-testid="episode-card"], article a, [role="link"]`) matched
  // NOTHING against the current Home — zero `article` elements, zero test-ids, zero role=link —
  // so this walk had been failing on its own stale DOM assumptions rather than on the app. An
  // href is the one thing that cannot drift while the surface still does its job.
  const firstEpisodeCard = page.locator('a[href*="/episode/"]').first()
  await expect(firstEpisodeCard).toBeVisible()
  await firstEpisodeCard.click()
  await page.waitForLoadState('networkidle')
  // Player surface is up (title + a transcript segment or the pending state).
  // Play is asserted STRICTLY: the Tier-3 stack contract includes a media host on :18765 (see
  // this suite's config header), so a missing Play button means the player failed to build a
  // transport for audio that IS being served — a real regression, not a stack gap.
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
  // Asserted: "Mark this moment" is present on the episode surface both before and during
  // playback (verified against this corpus), so a conditional click here only ever hid a
  // regression in the capture affordance.
  const markMoment = page.getByRole('button', { name: 'Mark this moment' })
  await expect(markMoment).toBeVisible()
  await markMoment.click()
  await page.screenshot({ path: 'validation-results/05-captured.png', fullPage: true })

  // === Verify in Library → Highlights ======================================
  await page.goto('/library')
  await page.waitForLoadState('networkidle')
  await page.screenshot({ path: 'validation-results/06-library.png', fullPage: true })

  // There is no "Highlights" BUTTON — Library's tabs are Following / Saved / Collections /
  // Revisit, and Highlights is a heading INSIDE the Saved tab. The old locator therefore never
  // matched, and because the click was conditional this walk silently stopped at the Library
  // shell: the capture made above was never actually verified as having landed. That is the
  // entire point of the walk, and it had not run once.
  await page.getByRole('button', { name: 'Saved', exact: true }).click()
  await page.waitForLoadState('networkidle')
  await expect(page.getByText('Highlights', { exact: true }).first()).toBeVisible()
  await page.screenshot({
    path: 'validation-results/07-highlights.png',
    fullPage: true,
  })
})
