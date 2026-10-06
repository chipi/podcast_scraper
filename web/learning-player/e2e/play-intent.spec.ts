import { expect, test, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/** A fresh account per RUN: these tests follow, save and delete, and an account reused from the
 *  last run starts in the state that run left (already Following) — a flip test then flips it off. */
const RUN = Date.now().toString(36)

/**
 * WHO starts audio (operator 2026-10-05): "tap = open and play just for Resume and where we have an
 * explicit play icon". Every other way into an episode — a What's new card or row, any episode
 * card — OPENS it, paused.
 *
 * Measured as calls to `HTMLMediaElement.play()`, not as `audio.paused`. A browser's autoplay
 * policy may refuse a play that no gesture on THAT page started, and the store swallows the
 * refusal (`NotAllowedError`) by design — so `paused` would answer "what did the browser allow",
 * while the rule is about what the APP asks for. Counting the asks answers the rule exactly, in
 * either policy.
 */
const EPISODE = 'Index Investing Without the Myths'

/** Count every play() the page makes, from before any script runs. */
async function countPlays(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const w = window as unknown as { __plays: number }
    w.__plays = 0
    const real = HTMLMediaElement.prototype.play
    HTMLMediaElement.prototype.play = function (...args) {
      w.__plays += 1
      return real.apply(this, args)
    }
  })
}
const plays = (page: Page) => page.evaluate(() => (window as unknown as { __plays: number }).__plays)

/** The episode's slug, from its card on the show page. */
async function episodeSlug(page: Page): Promise<string> {
  const eps = await (await page.request.get('/api/app/podcasts/p05/episodes')).json()
  return (eps as { items: { slug: string; title: string }[] }).items.find((e) =>
    e.title.startsWith('Index Investing'),
  )!.slug
}

/** The player has loaded this episode's audio (the element exists and has a source). */
async function audioArmed(page: Page): Promise<void> {
  await page.locator('audio').waitFor({ state: 'attached' })
  await expect
    .poll(() => page.evaluate(() => !!(document.querySelector('audio') as HTMLAudioElement)?.src))
    .toBe(true)
}

test.beforeEach(async ({ page }, testInfo) => {
  await countPlays(page)
  await signInIsolated(page, `play-intent-${testInfo.title.slice(0, 24)}-${RUN}`, testInfo)
})

test('opening an episode from its card does NOT start audio', async ({ page }) => {
  await page.goto('/podcast/p05')
  await page.getByText(EPISODE).first().click()
  await expect(page).toHaveURL(/\/episode\//)
  expect(new URL(page.url()).searchParams.get('play')).toBeNull()
  await audioArmed(page)
  // Give a would-be autoplay its chance before asserting it never came.
  await page.waitForTimeout(1500)
  expect(await plays(page), 'opening an episode started audio unprompted').toBe(0)
})

test("What's new: the #01 card and a 02+ row OPEN the episode, paused", async ({ page }) => {
  for (const pick of ['card', 'row'] as const) {
    await page.goto('/')
    const section = page.getByRole('heading', { name: "What's new" }).locator('xpath=ancestor::section[1]')
    await expect(section).toBeVisible()
    // #01 is the first episode link in the section; a ranked row is any later one.
    const links = section.locator('a[href^="/episode/"]')
    await expect(links.nth(1)).toBeVisible()
    const link = pick === 'card' ? links.first() : links.nth(1)
    const href = (await link.getAttribute('href')) ?? ''
    expect(href, `the What's new ${pick} carries a play intent`).not.toContain('play=1')
    const before = await plays(page)
    await link.click()
    await expect(page).toHaveURL(/\/episode\//)
    await audioArmed(page)
    await page.waitForTimeout(1500)
    expect(await plays(page), `the What's new ${pick} started audio`).toBe(before)
  }
})

test('an explicit "?t=…&play=1" link plays from the moment; "?t=" alone opens there paused', async ({
  page,
}) => {
  const slug = await episodeSlug(page)

  await page.goto(`/episode/${slug}?t=42`)
  await audioArmed(page)
  await expect
    .poll(() => page.evaluate(() => (document.querySelector('audio') as HTMLAudioElement).currentTime))
    .toBeGreaterThan(40)
  await page.waitForTimeout(1500)
  expect(await plays(page), '?t= alone must not start audio').toBe(0)

  await page.goto(`/episode/${slug}?t=65&play=1`)
  await audioArmed(page)
  await expect
    .poll(() => page.evaluate(() => (document.querySelector('audio') as HTMLAudioElement).currentTime))
    .toBeGreaterThan(60)
  await expect.poll(() => plays(page), { message: '?play=1 never asked to play' }).toBeGreaterThan(0)
})

test('a "Play from" link to the episode ALREADY OPEN still seeks and plays (same page)', async ({
  page,
}) => {
  // The start position applies once per load; a link to the same episode reuses the page. It used
  // to do nothing — no seek, no play (73e6483f1).
  const slug = await episodeSlug(page)
  await page.goto(`/episode/${slug}`)
  await audioArmed(page)
  const before = await plays(page)
  // In-app navigation (the router), as a topic card's "▶ Play from" on this page does.
  await page.evaluate((s) => {
    const app = document.querySelector('#app') as unknown as {
      __vue_app__: { config: { globalProperties: { $router: { push: (to: string) => void } } } }
    }
    app.__vue_app__.config.globalProperties.$router.push(`/episode/${s}?t=30&play=1`)
  }, slug)
  await expect(page).toHaveURL(/t=30/)
  await expect
    .poll(() => page.evaluate(() => (document.querySelector('audio') as HTMLAudioElement).currentTime))
    .toBeGreaterThan(28)
  await expect.poll(() => plays(page)).toBeGreaterThan(before)
})

test('Search: "▶ Play from" carries play=1 and starts the episode at the passage', async ({ page }) => {
  await page.goto('/search?q=index%20funds')
  const playFrom = page.getByTestId('play-from').first()
  await expect(playFrom).toBeVisible({ timeout: 30_000 })
  await playFrom.click()
  await expect(page).toHaveURL(/\/episode\/.*[?&]play=1/)
  await expect(page).toHaveURL(/[?&]t=\d+/)
  await audioArmed(page)
  await expect.poll(() => plays(page), { message: 'Play from opened the episode paused' }).toBeGreaterThan(0)
})

test('Saved: a highlight\'s "▶ Play from" carries play=1', async ({ page }) => {
  const slug = await episodeSlug(page)
  const r = await page.request.post('/api/app/highlights', {
    data: { episode_slug: slug, kind: 'span', start_ms: 65_000, quote_text: 'a line worth keeping' },
  })
  expect(r.ok()).toBe(true)
  await page.goto('/library?tab=saved')
  const jump = page.getByTestId('highlight-card').getByTestId('play-from').first()
  await expect(jump).toBeVisible()
  await expect(jump).toHaveAttribute('href', /[?&]t=65/)
  await expect(jump).toHaveAttribute('href', /[?&]play=1/)
  await jump.click()
  await expect(page).toHaveURL(/play=1/)
  await audioArmed(page)
  await expect.poll(() => plays(page)).toBeGreaterThan(0)
})
