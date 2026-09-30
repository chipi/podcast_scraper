import { expect, test } from '@playwright/test'
import { navTo, signInIsolated } from './helpers'

/**
 * Where the queue lives, and how you get to it.
 *
 * #1838 moved Up next / Recently played out of the Library tabs into a bottom-sheet panel opened
 * from the transport — first from both players, then (2026-09-23) from the full player only, and
 * now from neither: the operator removed the last opener on 2026-09-27 because it crowded the one
 * row whose job is playback, and the masthead already carries a queue control at every width and on
 * every screen. `QueuePanel` is deleted; `/queue` (`QueueView`) is the queue's one surface and it
 * renders BOTH halves.
 *
 * So what this file proves is the ROUTE, not a modal: the masthead reaches the queue from a page
 * that is not the player, both sections are there on arrival, and neither removed opener has
 * quietly come back. The rows themselves are covered by queue-reorder.spec.
 */

test('the masthead reaches the queue, and the destination carries both halves', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'queue-panel-mini', testInfo)
  // Start playback so the mini-player is present, then leave the player in-app.
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  // WAIT FOR THE FIRST POSITION SAVE TO LAND. "Recently played" is fetched once, when /queue mounts,
  // and this episode is only in it once its `PUT /playback` has been accepted. The first save leaves
  // on the first `timeupdate`, so navigating as soon as audio moves raced it: the list painted the
  // empty state and never refreshed (flaked 2026-09-29, `queue-panel-recent` not found in 15s while
  // the "Recently played" heading was on screen).
  const firstSave = page.waitForResponse(
    (r) => r.request().method() === 'PUT' && /\/playback\//.test(r.url()) && r.ok(),
    { timeout: 20_000 },
  )
  await page.getByRole('button', { name: 'Play', exact: true }).first().click()
  await expect
    .poll(async () => page.evaluate(() => document.querySelector('audio')?.currentTime ?? 0), {
      timeout: 15_000,
    })
    .toBeGreaterThan(0.2)
  await firstSave

  await navTo(page, 'search')
  await expect(page.getByTestId('mini-player')).toBeVisible()

  /*
   * The mini-player's queue button is GONE (operator 2026-09-23). The masthead now carries a queue
   * control at every width, so the bar kept a second route to the same place on a screen that
   * already showed the first — and the bar is the most space-constrained strip in the app.
   *
   * Asserted as an ABSENCE as well as a replacement: a spec that only checks the new path would
   * stay green if the old button quietly came back.
   */
  await expect(page.getByTestId('mini-player-queue')).toHaveCount(0)
  // What the bar carries instead: save and collect, for the episode that is playing.
  await expect(page.getByTestId('mini-player').getByRole('button', { name: /save|remove/i }).first()).toBeVisible()

  await page.getByTestId('masthead-queue').click()
  await expect(page).toHaveURL(/\/queue/)
  // The destination shows BOTH halves — Up next and the history — because it is now the queue's
  // primary surface, not a subset of a panel reached from the player.
  await expect(page.getByRole('heading', { name: 'Up next' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Recently played' })).toBeVisible()
  await expect(page.getByTestId('queue-panel-recent')).toBeVisible()
})

test('the full player queues THIS episode instead of opening a panel', async ({
  page,
}, testInfo) => {
  /*
   * The opener that used to sit in the transport row is gone (operator 2026-09-27), and what took
   * its place beside the heart is a different control: it acts on the episode rather than
   * navigating, and it SHOWS whether this episode is already queued — which an open-a-panel button
   * structurally could not.
   *
   * Both halves are asserted. Checking only that the new toggle works would stay green if the old
   * opener came back beside it, which is the arrangement the operator objected to.
   */
  await signInIsolated(page, 'queue-panel-player', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  await expect(page.getByTestId('player-queue')).toHaveCount(0)
  await expect(page.getByTestId('queue-panel')).toHaveCount(0)

  // START FROM AN EMPTY QUEUE, set through the API before the assertions. The account is per
  // spec+project, not per attempt, so a retry — or any second run against the same api — finds this
  // episode already queued. Checking the button for that and un-queueing it was RACY (2026-09-30):
  // the button renders "Add to queue" until the queue store hydrates, so the check read "not
  // queued", then the store landed, the button flipped to "Remove", and "Add" never appeared.
  const reset = await page.request.put('/api/app/queue', { data: { items: [] } })
  expect(reset.ok(), `PUT /api/app/queue → ${reset.status()}`).toBe(true)
  await page.reload()

  // SCOPED to the player's own action row. The page also renders related-episode rows, each with
  // its own queue toggle, and on desktop they load in while this runs — measured 2026-09-30 as 7
  // "Add to queue" buttons, a strict-mode violation, on a run where the related rows won the race.
  const actions = page.getByTestId('player-actions')

  // The heart's row owns it now. Named by its action, and the name flips once the write lands.
  const add = actions.getByRole('button', { name: 'Add to queue' })
  await expect(add).toBeVisible()
  await add.click()
  await expect(actions.getByRole('button', { name: 'Remove from queue' })).toBeVisible()

  // ...and the episode really is in the queue, not merely relabelled optimistically.
  await page.getByTestId('masthead-queue').click()
  await expect(page).toHaveURL(/\/queue/)
  await expect(page.getByText('Index Investing Without the Myths').first()).toBeVisible()
})
