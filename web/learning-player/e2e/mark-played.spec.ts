import { expect, test, type Page } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * "Mark as played" where things wait to be heard (operator 2026-10-14): beside Resume on Continue
 * listening, in the ⋯ of an In progress episode card (Jump back in until 2026-10-14), and in the ⋯ of an Up next card. Marking takes the
 * episode out of where it is at once, and out of the server's in-progress list and the queue, so a
 * reload agrees. Through the real app and API: positions and the queue are written over HTTP the
 * way the player writes them.
 */

async function episodes(page: Page, n: number): Promise<string[]> {
  const r = await page.request.get(`/api/app/episodes?page_size=${n}`)
  expect(r.ok()).toBe(true)
  return ((await r.json()).items as Array<{ slug: string }>).map((e) => e.slug)
}

/**
 * Two listens, the second newer. Positions stay well short of the ~4-minute fixture episodes, so
 * neither reads as finished; a second apart, so "newest" is not a tie inside one timestamp.
 */
async function startTwo(page: Page, older: string, newer: string): Promise<void> {
  for (const [slug, seconds] of [
    [older, 30],
    [newer, 60],
  ] as const) {
    const r = await page.request.put(`/api/app/playback/${encodeURIComponent(slug)}`, {
      data: { position_seconds: seconds, finished: false },
    })
    expect(r.ok()).toBe(true)
    await page.waitForTimeout(1100)
  }
}

/** The episode the resume card is showing, from its own link. */
async function resumeSlug(page: Page): Promise<string> {
  const href = await page.getByTestId('home-resume-open').getAttribute('href')
  return decodeURIComponent(href!.split('/').filter(Boolean).pop()!.split('?')[0]!)
}

async function inProgress(page: Page): Promise<string[]> {
  const r = await page.request.get('/api/app/playback?in_progress=true&limit=20')
  return ((await r.json()).items as Array<{ slug: string }>).map((p) => p.slug)
}

async function home(page: Page): Promise<void> {
  await page.goto('/')
  for (const name of ['Not now', 'Skip step']) {
    const b = page.getByRole('button', { name, exact: true })
    if (await b.isVisible().catch(() => false)) await b.click().catch(() => {})
  }
}

test('Continue listening: Mark as played beside Resume moves the card on, and a reload agrees', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'mark-played-resume', testInfo)
  const [older, newest] = await episodes(page, 2)
  await startTwo(page, older!, newest!)
  await home(page)

  // Wait for the newest listen on the card: until both are loaded the card can show the other one,
  // and reading it early marked the wrong episode (2026-10-14).
  await expect.poll(() => resumeSlug(page)).toBe(newest)
  const marked = newest!
  const kept = older!
  const mark = page.getByTestId('home-resume-mark-played')
  await expect(mark).toBeVisible()
  // Both pills on one line of text each, side by side, inside the screen — down to a 320 px phone.
  const resume = page.getByTestId('home-resume')
  for (const width of [page.viewportSize()!.width, 320]) {
    await page.setViewportSize({ width, height: page.viewportSize()!.height })
    const [r, m] = [(await resume.boundingBox())!, (await mark.boundingBox())!]
    const row = (await resume.locator('..').boundingBox())!
    expect(r.height).toBeLessThanOrEqual(48)
    expect(m.height).toBeLessThanOrEqual(48)
    expect(Math.abs(r.y - m.y)).toBeLessThan(2)
    expect(m.x).toBeGreaterThan(r.x + r.width - 1)
    // Inside the row, not just the screen: the card clips what overflows it ("Mark played" was cut
    // off at 320 while still inside the screen).
    expect(m.x + m.width).toBeLessThanOrEqual(row.x + row.width + 0.5)
    if (process.env.MARK_PLAYED_SHOTS) {
      await page.screenshot({ path: `${process.env.MARK_PLAYED_SHOTS}/resume-${testInfo.project.name}-${width}.png` })
    }
  }
  await mark.click()
  await expect.poll(() => resumeSlug(page)).toBe(kept)
  await expect.poll(() => inProgress(page)).toEqual([kept])

  await page.reload()
  await expect.poll(() => resumeSlug(page)).toBe(kept)
})

test('In progress: Mark as played in an episode card ⋯ takes the card out of the rail', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'mark-played-jump-back', testInfo)
  const [first, second] = await episodes(page, 2)
  await startTwo(page, first!, second!)
  await home(page)
  await expect.poll(() => resumeSlug(page)).toBe(second)
  const onCard = second!

  // "Jump back in" is folded into In progress (operator 2026-10-14): the other episode is a card there.
  const rail = page.getByTestId('in-progress-episode')
  await expect(rail).toHaveCount(1)
  await rail.getByRole('button', { name: /^More actions/ }).first().click()
  await page.getByTestId('episode-mark-played').click()
  await expect(rail).toHaveCount(0)
  await expect.poll(() => inProgress(page)).toEqual([onCard])
  await expect.poll(() => resumeSlug(page)).toBe(onCard)
})

test('Up next: Mark as played in a card ⋯ takes it out of the queue', async ({ page }, testInfo) => {
  await signInIsolated(page, 'mark-played-queue', testInfo)
  const [queued, other] = await episodes(page, 2)
  for (const slug of [queued, other]) {
    const r = await page.request.post('/api/app/queue/items', { data: { slug, after: null } })
    expect(r.ok()).toBe(true)
  }
  await page.goto('/queue')
  const menus = page.getByRole('button', { name: /^More actions/ })
  await expect(menus.first()).toBeVisible()
  const queueNow = async (): Promise<string[]> =>
    (await (await page.request.get('/api/app/queue')).json()).items as string[]
  const before = await queueNow()
  await menus.first().click()
  await page.getByTestId('episode-mark-played').click()
  await expect.poll(queueNow).toEqual(before.slice(1))
  await expect(page.getByRole('button', { name: /^More actions/ })).toHaveCount(before.length - 1)
})
