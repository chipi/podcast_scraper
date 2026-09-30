import { expect, test } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * Episode notes panel — the people in the room lead the panel, with their photos.
 *
 * Real backend, real corpus: `/episodes/{slug}/entities` now attaches the hosted-photo route from
 * `enrichments/person_web.json`, the same as Top voices. The panel episode "The Risk Panel"
 * (p05_e04) has a host and two guests.
 *
 * The photo assertion checks the image actually DECODED (naturalWidth > 0), not just that an <img>
 * exists: a relative photo route that 404s still renders an <img> for a moment before
 * ProfileAvatar falls back to initials, and that silent fallback is how this bug has shipped before.
 */
test('host and guests lead the panel with photos, and open in the panel with a Back', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'episode-notes-people', testInfo)
  await page.goto('/')
  await page.goto('/podcast/p05')
  await page.getByText('The Risk Panel: Diversify or Concentrate?').first().click()
  await page.getByTestId('player-open-insights').click()

  const people = page.getByTestId('kp-dossier-person')
  await expect(people).toHaveCount(3)
  await expect(people.first()).toContainText('Host')

  const photo = people.filter({ hasText: 'Daniel Cho' }).locator('img')
  await expect(photo).toBeVisible()
  await expect
    .poll(() => photo.evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0))
    .toBe(true)

  // Same path as the person chip: replace-in-panel, with a Back that returns to the notes.
  await people.filter({ hasText: 'Daniel Cho' }).click()
  const back = page.getByTestId('ec-dismiss')
  await expect(back).toHaveAttribute('aria-label', 'Back')
  await expect(page.getByTestId('kp-episode-dossier')).toHaveCount(0)
  await back.click()
  await expect(page.getByTestId('kp-episode-dossier')).toBeVisible()
})
