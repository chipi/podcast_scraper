import { expect, test } from '@playwright/test'
import { routeLoadableAudio } from './helpers'

/**
 * Audio survives navigation (#1587) — the property the whole change exists for.
 *
 * The `<audio>` element used to be rendered by PlayerView, so leaving an episode unmounted it and
 * playback stopped. You could not browse, search or look anything up while listening — and the
 * learning features are most valuable mid-listen, so every one of those journeys cost you the thing
 * you were listening to. This asserts the fix at the only level that proves it: a real browser,
 * across a real route change.
 */

test('playback continues across navigation, and the mini-player offers the way back', async ({
  page,
}) => {
  await routeLoadableAudio(page) // before the first navigation — the fixture MP3 is undecodable

  // Reach the episode via its show page — date-independent, same route the other specs use.
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  // Start playback and confirm the element is actually advancing, not merely "not erroring".
  await page.getByRole('button', { name: 'Play', exact: true }).first().click()
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.currentTime ?? 0), {
      timeout: 15_000,
    })
    .toBeGreaterThan(0.2)

  const episodeUrl = page.url()

  // No mini-player on the player page itself — the full transport is right there.
  await expect(page.getByTestId('mini-player')).toHaveCount(0)

  // Navigate away IN-APP. Must be a router navigation, not page.goto — a full reload tears down
  // the whole SPA and no amount of store ownership survives that. Client-side navigation is the
  // case that was broken: the view unmounted and took the <audio> element with it.
  // The HEADER nav, not the bottom bar: the bottom bar is sm:hidden, so clicking it fails on
  // desktop-chrome. Same dual-viewport trap `openTranscript` exists for.
  await page.locator('header').getByRole('link', { name: 'Search' }).click()
  await expect(page).toHaveURL(/\/search/)
  await expect(page.getByTestId('mini-player')).toBeVisible()

  const atNav = await page.evaluate(() => document.querySelector('audio')?.currentTime ?? 0)
  expect(atNav, 'audio element must still exist after leaving the player').toBeGreaterThan(0)

  // Still ADVANCING, not just present — a paused-but-alive element would be a different bug.
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.currentTime ?? 0), {
      timeout: 15_000,
    })
    .toBeGreaterThan(atNav)

  // The mini-player is the way back to what is playing.
  await page.getByTestId('mini-player-open').click()
  await expect(page).toHaveURL(episodeUrl)
  await expect(page.getByTestId('mini-player')).toHaveCount(0)
})

test('the mini-player pauses and resumes from anywhere', async ({ page }) => {
  await routeLoadableAudio(page)

  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await page.getByRole('button', { name: 'Play', exact: true }).first().click()
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.currentTime ?? 0), {
      timeout: 15_000,
    })
    .toBeGreaterThan(0.2)

  await page.locator('header').getByRole('link', { name: 'Browse' }).click()
  const toggle = page.getByTestId('mini-player-toggle')
  await expect(toggle).toBeVisible()

  await toggle.click()
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.paused ?? true))
    .toBe(true)

  await toggle.click()
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.paused ?? true))
    .toBe(false)
})
