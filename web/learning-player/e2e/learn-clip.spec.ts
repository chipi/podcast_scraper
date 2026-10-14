import { expect, test, type Page } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * A learning day's clip closes the loop (operator 2026-10-14): Play opens the day's episode playing
 * its clip and stops at the clip's end, the day is done by having heard it, and the end card goes
 * back to the week (or on into the episode). "Play all" plays every day still to do, back to back
 * across episodes, and a stop leaves the week where it is.
 *
 * Through the real app, API and audio: "Building trails that last" is four days of six-second
 * clips on four episodes, short enough to hear in full.
 */
const SUBJECT = 'thc:building-trails'

async function startPath(page: Page): Promise<void> {
  const r = await page.request.post('/api/app/discover/learn', { data: { subject_id: SUBJECT } })
  expect(r.ok()).toBe(true)
}

async function daysDone(page: Page): Promise<number[]> {
  const r = await page.request.get(`/api/app/discover/learn/paths/${encodeURIComponent(SUBJECT)}`)
  return ((await r.json()).steps as Array<{ day: number; done: boolean }>).filter((s) => s.done).map((s) => s.day)
}

test('a day\'s Play stops at the clip\'s end, does the day, and goes back to the week', async ({ page }, testInfo) => {
  await signInIsolated(page, 'learn-clip-day', testInfo)
  await startPath(page)
  await page.goto(`/learn?subject=${encodeURIComponent(SUBJECT)}`)
  const today = page.locator('[data-testid="learn-step"][data-open="true"]')
  await today.getByTestId('learn-step-play').click()

  await expect(page).toHaveURL(/\/episode\/[^/?]+\?learn=thc(:|%3A)building-trails&day=1&from=learn$/)
  await expect(page.getByTestId('learn-clip-kicker')).toHaveText(/^Learn · day 1 · \d+:\d\d$/)
  const end = page.getByTestId('learn-clip-done')
  await expect(end).toHaveAttribute('data-status', 'done', { timeout: 30_000 })
  await expect(end.getByTestId('learn-clip-kicker')).toHaveText('Day 1 done ✓')
  // Stopped at the clip's end, not playing on into the episode.
  expect(await page.evaluate(() => document.querySelector('audio')!.paused)).toBe(true)
  expect(await daysDone(page)).toEqual([1])

  await end.getByTestId('learn-clip-back').click()
  await expect(page).toHaveURL(/\/learn\?subject=thc(:|%3A)building-trails$/)
  const day1 = page.locator('[data-testid="learn-step"][data-day="1"]')
  await expect(day1.getByTestId('learn-step-done-mark')).toBeVisible()
  await expect(page.getByTestId('learn-step-next')).toBeVisible()
})

test('"Keep listening" on the end card carries on into the episode as a listen', async ({ page }, testInfo) => {
  await signInIsolated(page, 'learn-clip-keep', testInfo)
  await startPath(page)
  await page.goto(`/learn?subject=${encodeURIComponent(SUBJECT)}`)
  await page.locator('[data-testid="learn-step"][data-open="true"]').getByTestId('learn-step-play').click()
  const end = page.getByTestId('learn-clip-done')
  await expect(end).toHaveAttribute('data-status', 'done', { timeout: 30_000 })
  await end.getByTestId('learn-clip-keep').click()
  await expect(page).not.toHaveURL(/learn=/)
  await expect(end).toHaveCount(0)
  await expect.poll(() => page.evaluate(() => document.querySelector('audio')!.paused)).toBe(false)
})

test('"Play all" plays every day back to back and does each one', async ({ page }, testInfo) => {
  test.setTimeout(120_000)
  await signInIsolated(page, 'learn-clip-all', testInfo)
  await startPath(page)
  await page.goto(`/learn?subject=${encodeURIComponent(SUBJECT)}`)
  await page.getByTestId('learn-play-all').click()
  await expect(page.getByTestId('learn-play-all-toggle')).toContainText('Day 1')
  await expect.poll(() => daysDone(page), { timeout: 90_000 }).toEqual([1, 2, 3, 4])
  // The week shows it without a reload, and there is nothing left to play.
  await expect(page.getByTestId('learn-step-done-mark')).toHaveCount(4)
  await expect(page.getByTestId('learn-play-all')).toHaveCount(0)
  await expect(page.getByTestId('learn-view-finished')).toHaveText('All 4 days done ✓')

  // Finished (operator 2026-10-14): off Home's In progress at once, kept as history in Stats.
  const inProgress = await (await page.request.get('/api/app/discover/in-progress')).json()
  expect(inProgress.paths).toEqual([])
  await page.goto('/')
  await page.waitForLoadState('networkidle')
  await expect(page.getByTestId('in-progress-learn')).toHaveCount(0)
  await page.goto('/profile?tab=stats')
  await expect(page.getByTestId('stats-learning-history-item').filter({ hasText: 'Building trails that last' })).toContainText(
    '4 days ·',
  )
})

test('a stop in "Play all" leaves the week where it is; the next one starts at the first day not done', async ({ page }, testInfo) => {
  test.setTimeout(90_000)
  await signInIsolated(page, 'learn-clip-resume', testInfo)
  await startPath(page)
  await page.goto(`/learn?subject=${encodeURIComponent(SUBJECT)}`)
  await expect(page.getByTestId('learn-play-all')).toHaveText('▶ Play all')
  await page.getByTestId('learn-play-all').click()
  await expect.poll(() => daysDone(page), { timeout: 45_000 }).toEqual([1])
  await page.getByTestId('learn-play-all-stop').click()
  await expect(page.getByTestId('learn-play-all-live')).toHaveCount(0)

  await page.reload()
  await expect(page.getByTestId('learn-play-all')).toHaveText('▶ Play all from day 2')
  expect(await daysDone(page)).toEqual([1])
})
