import { expect, test, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Finishing an episode OFFLINE, end to end (operator 2026-09-23).
 *
 * Reported from a flight: two episodes finished in airplane mode, a third after landing. Back
 * online, none of them read as played anywhere in the app — "Jump back in" still offered them, the
 * catalogue's Played filter matched nothing, and no list showed any marker.
 *
 * Two independent failures sat behind that, and these pin both:
 *
 *  1. "Played" had TWO records and the UI read one. `completed` was the hand-marked list; the
 *     finish the player itself recorded lived in `finished_at` and reached no surface — so
 *     finishing by LISTENING, the normal way, marked nothing anywhere. `GET /api/app/completed`
 *     now merges them.
 *  2. A finish written offline has to survive the flight: it rides the position outbox and is
 *     pushed on reconnect, stamped with when the listener was actually there (#1913).
 *
 * The finish is dispatched as an `ended` event on the `<audio>` element, the same trigger
 * `recap-panel.spec.ts` uses: headless Chromium will not reliably decode the fixture audio, and the
 * store's `ended` handler is the code path under test either way.
 *
 * Every assertion that matters is made against the SERVER's answer or a reloaded page, so a pass
 * means the state round-tripped — not that an in-memory store still remembers being told.
 */

test.use({ serviceWorkers: 'allow' })

const SHOW = '/podcast/p05'
const EPISODE = 'Index Investing Without the Myths'

/**
 * A fresh user per RUN, not per spec name.
 *
 * `signInIsolated` derives its id from the name you pass, so two runs of this file share an
 * account — and the API's data dir outlives the run. The second run therefore asserted against the
 * FIRST run's state: three of these "passed" in under 500ms without doing any of the work. Every
 * test below also asserts the episode is NOT played before it acts, so a leaked account fails loudly
 * instead of passing silently.
 */
const RUN = Date.now().toString(36)

/** Force the current episode to finish. */
async function finishEpisode(page: Page): Promise<void> {
  await page.locator('[data-testid="app-audio"]').waitFor({ state: 'attached', timeout: 15_000 })
  await page.evaluate(() => document.querySelector('audio')?.dispatchEvent(new Event('ended')))
}

/** Open a known episode in the player and return its slug (the route is `/episode/:slug`). */
async function openEpisode(page: Page): Promise<string> {
  await page.goto(SHOW)
  await page.getByText(EPISODE).first().click()
  await expect(page).toHaveURL(/\/episode\//)
  return new URL(page.url()).pathname.replace('/episode/', '')
}

/** What the SERVER says is played, asked directly rather than inferred from the DOM. */
async function serverCompleted(page: Page): Promise<string[]> {
  return page.evaluate(async () => {
    const res = await fetch('/api/app/completed', { credentials: 'include' })
    if (!res.ok) return []
    return (await res.json()).slugs as string[]
  })
}

test('finished OFFLINE: it syncs on reconnect, and the app then calls it played', async ({
  page,
  context,
}, testInfo) => {
  await signInIsolated(page, `offline-finish-${RUN}`, testInfo)
  const slug = await openEpisode(page)
  expect(await serverCompleted(page), 'precondition: this account has not played it').not.toContain(slug)

  // Airplane mode, then listen to the end. The position PUT cannot leave the device.
  await context.setOffline(true)
  await finishEpisode(page)

  // Land.
  await context.setOffline(false)
  await page.evaluate(() => window.dispatchEvent(new Event('online')))

  // The outbox flush is what carries it across; poll the server rather than reload-and-hope, so a
  // failure reads as "it never synced" instead of "the reload was too early".
  await expect
    .poll(() => serverCompleted(page), {
      timeout: 30_000,
      message: 'the offline finish never reached the server after reconnect',
    })
    .toContain(slug)

  // And now the surface the operator actually looked at. Reloaded, so this is server truth.
  await page.goto('/catalog')
  await page.getByTestId('list-toolbar-filter').click()
  await page.getByTestId('list-toolbar-filter-opt-played').click()
  await expect(
    page.locator('article').filter({ has: page.locator(`a[href="/episode/${slug}"]`) }),
  ).toBeVisible({ timeout: 20_000 })
})

test('started offline, finished back ONLINE: same answer', async ({ page, context }, testInfo) => {
  // The second case from the report: begun in the air, ran out after landing. The interesting part
  // is that a pending offline POSITION for this episode is already queued when the online finish
  // lands — the flush must not overwrite the newer finish with the older stub.
  await signInIsolated(page, `offline-start-${RUN}`, testInfo)
  const slug = await openEpisode(page)
  expect(await serverCompleted(page), 'precondition: this account has not played it').not.toContain(slug)

  await context.setOffline(true)
  // Partway, offline: recorded on the device, pending, and NOT played.
  await page.evaluate(() => {
    const audio = document.querySelector('audio') as HTMLAudioElement | null
    if (!audio) return
    audio.currentTime = 30
    audio.dispatchEvent(new Event('timeupdate'))
  })

  await context.setOffline(false)
  await page.evaluate(() => window.dispatchEvent(new Event('online')))
  await finishEpisode(page)

  await expect
    .poll(() => serverCompleted(page), {
      timeout: 30_000,
      message: 'finishing online after an offline start did not mark it played',
    })
    .toContain(slug)

  // It must also LEAVE "Jump back in": that rail is for episodes with something left to resume, and
  // one that ran out has nothing. This was the visible half of the report.
  await page.goto('/')
  await expect
    .poll(
      () => page.locator(`[data-testid="home-jump-back-in"] a[href="/episode/${slug}"]`).count(),
      { timeout: 20_000, message: 'a finished episode is still offered in Jump back in' },
    )
    .toBe(0)
})

test('a played episode carries its marker in the catalogue, after a reload', async ({
  page,
}, testInfo) => {
  // The third complaint: no list said an episode had been heard. Asserted after navigating away and
  // back, so the badge is driven by server state rather than by the toggle that set it.
  await signInIsolated(page, `played-badge-${RUN}`, testInfo)
  const slug = await openEpisode(page)
  expect(await serverCompleted(page), 'precondition: this account has not played it').not.toContain(slug)
  await finishEpisode(page)

  await expect.poll(() => serverCompleted(page), { timeout: 30_000 }).toContain(slug)

  // The SHOW page, not the catalogue: the catalogue paginates and this episode need not be on the
  // first page, which would fail as "no badge" while actually meaning "no card".
  await page.goto(SHOW)
  const card = page.locator('article').filter({ has: page.locator(`a[href="/episode/${slug}"]`) })
  await expect(card).toBeVisible({ timeout: 20_000 })
  await expect(card.getByTestId('played-badge')).toBeVisible({ timeout: 20_000 })
})

test('mark-unplayed takes a FINISHED episode back off the played list', async ({
  page,
}, testInfo) => {
  // The toggle has to come back. With `/completed` merging the finish record, retracting only the
  // hand-marked half would leave it played forever — tap "Mark as unplayed", and it returns.
  await signInIsolated(page, `unplay-${RUN}`, testInfo)
  const slug = await openEpisode(page)
  expect(await serverCompleted(page), 'precondition: this account has not played it').not.toContain(slug)
  await finishEpisode(page)
  await expect.poll(() => serverCompleted(page), { timeout: 30_000 }).toContain(slug)

  await page.goto(`/episode/${slug}`)
  // The player masthead's ⋯ — mark-as-played is a deliberate, secondary action and lives there.
  await page.getByTestId('overflow-trigger').first().click()
  const item = page.getByTestId('mark-played')
  await expect(item).toHaveText(/unplayed/i)
  await item.click()

  await expect
    .poll(() => serverCompleted(page), {
      timeout: 30_000,
      message: 'mark-unplayed did not retract the finish, so the toggle only goes one way',
    })
    .not.toContain(slug)
})
