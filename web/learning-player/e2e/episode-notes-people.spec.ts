import { expect, test } from '@playwright/test'

import { signInIsolated, tapAndRecordTop } from './helpers'

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
  // The crop is anchored near the top, not centred (UXS-011 ProfileAvatar: a centred crop took the
  // top of the head off most real portraits). Checked on the RESOLVED style, not the class.
  expect(await photo.evaluate((img) => getComputedStyle(img).objectPosition)).toBe('50% 10%')

  // Same path as the person chip: replace-in-panel, with a Back that returns to the notes.
  await people.filter({ hasText: 'Daniel Cho' }).click()
  const back = page.getByTestId('ec-dismiss')
  await expect(back).toHaveAttribute('aria-label', 'Back')
  await expect(page.getByTestId('kp-episode-dossier')).toHaveCount(0)
  await back.click()
  await expect(page.getByTestId('kp-episode-dossier')).toBeVisible()
})

/**
 * The panel is the episode's notes: one labelled way in, its own title, and an order that opens on
 * what the episode IS before offering to dig into it (UXS-011 / UXS-014, 2026-09-30).
 *
 * Labels are asserted as text here on purpose. Every other spec opens the panel by testid so a
 * copy change cannot break them; this is the one place that proves the copy the user reads.
 */
test('Episode notes: one labelled entry, its own title, and the notes-first order', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'episode-notes-order', testInfo)
  await page.goto('/')
  await page.goto('/podcast/p05')
  await page.getByText('The Risk Panel: Diversify or Concentrate?').first().click()

  const opener = page.getByTestId('player-open-insights')
  await expect(opener).toHaveText(/Episode notes/)
  // The standalone Summary pill is gone: it opened a subset of this panel. By role and name, not by
  // its old testid — a selector for a testid the app no longer renders proves nothing.
  await expect(page.getByTestId('player-hero').getByRole('button', { name: /summary/i })).toHaveCount(0)

  await opener.click()
  const panel = page.getByTestId('knowledge-panel')
  await expect(panel.getByText('Episode notes', { exact: true }).first()).toBeVisible()
  await expect(panel.getByTestId('episode-notes-export')).toContainText('Download notes')

  const y = async (l: ReturnType<typeof panel.locator>) => (await l.boundingBox())!.y
  const order = [
    await y(panel.getByTestId('kp-episode-dossier')),
    await y(panel.getByRole('heading', { name: 'Summary', exact: true })),
    await y(panel.getByTestId('episode-notes-export')),
    await y(panel.locator('#kp-ask')),
    await y(panel.getByTestId('summary-bullets')),
  ]
  expect(
    order,
    'expected episode (title + people) → Summary → Download notes → Search → Key points, top to bottom',
  ).toEqual([...order].sort((a, b) => a - b))
})

/**
 * Back from a person or topic opened in the notes returns to the row it was opened from
 * (operator 2026-10-04). The card REPLACES the panel body, so the panel used to rebuild at the top.
 */
test('closing a card opened from the notes chips returns the panel to those chips', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 390, height: 760 })
  await signInIsolated(page, 'episode-notes-back-scroll', testInfo)
  await page.goto('/')
  await page.goto('/podcast/p05')
  await page.getByText('The Risk Panel: Diversify or Concentrate?').first().click()
  await page.getByTestId('player-open-insights').click()

  const chip = page.getByTestId('kp-topic-chip').last()
  await chip.scrollIntoViewIfNeeded()
  // The panel body is its own scroller (the page behind the sheet does not move).
  const panelScroll = () =>
    chip.evaluate((el) => {
      let n: HTMLElement | null = el.parentElement
      while (n && getComputedStyle(n).overflowY !== 'auto') n = n.parentElement
      return n?.scrollTop ?? -1
    })
  const before = await panelScroll()
  expect(before, 'the chips are not below the fold, so this proves nothing').toBeGreaterThan(100)

  const seenAt = await tapAndRecordTop(chip)
  await expect(page.getByTestId('kp-episode-dossier')).toHaveCount(0)
  await page.getByTestId('ec-dismiss').click()
  await expect(chip).toBeInViewport({ ratio: 1 })
  await expect
    .poll(async () => Math.round(Math.abs((await chip.boundingBox())!.y - seenAt)), {
      message: 'the chip is not back at the spot on screen it was tapped at',
    })
    .toBeLessThan(12)
})
