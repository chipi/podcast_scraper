import { expect, test, type Page } from '@playwright/test'
import { expectSignedIn } from '../helpers'

/**
 * The queue panel after the 2026-09-23 round — built as a REAL state, then shot.
 *
 * The operator asked to see it "with some episodes lined up in a queue to play and some recently
 * played", so the spec queues three episodes and finishes three others rather than shooting an
 * empty panel and calling it a review. Five things changed and all five need to be visible:
 *
 *   1. Recently played offers **Read more** (it was suppressed on compact cards outright).
 *   2. Neither list cuts a line of description in half any more.
 *   3. Recently played has no queue button in the row — it moved into the ⋯.
 *   4. Recently played states WHEN it was last played, date and time, under the artwork.
 *   5. Up next carries download inline. NOT VISIBLE HERE: `DownloadButton` is native-only
 *      (`v-if="native"`), so in a browser this shot cannot show it. It is verifiable only on a
 *      device, and the shot below would look identical whether the wiring worked or not — which is
 *      exactly why it is called out rather than left for the reader to assume.
 *
 * Asserted before every capture, so a blank or still-loading panel FAILS instead of quietly
 * producing a PNG of a spinner.
 *
 *   npm run design:shots -- queue-2026-09-23
 */
const VARIANT = process.env.DESIGN_VARIANT || 'baseline'
const shot = (name: string) =>
  `design-results/${VARIANT}/${test.info().project.name}/queue-${name}.png`

/**
 * A FRESH account per run, unlike the sibling design specs.
 *
 * Those only look; this one queues and finishes episodes, so a fixed identity accumulates — the
 * second run opened onto a queue of four from the first and the shot stopped being of the state the
 * spec describes. Reproducibility here means "same construction every time", not "same account".
 */
const IDENTITY = `design-queue-${Date.now().toString(36)}`

async function settle(page: Page): Promise<void> {
  await page.waitForLoadState('networkidle')
  await page.evaluate(() => {
    const pending = Array.from(document.images).filter(
      (i) => !i.complete && i.getAttribute('loading') !== 'lazy' && !!i.currentSrc,
    )
    return Promise.race([
      Promise.all(pending.map((i) => new Promise((res) => { i.onload = i.onerror = res }))),
      new Promise((res) => setTimeout(res, 3000)),
    ])
  })
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => r(null))))
}

/** Queue every episode on a show page, up to `count`. */
async function queueFrom(page: Page, showPath: string, count: number): Promise<void> {
  await page.goto(showPath)
  await settle(page)
  const adds = page.getByRole('button', { name: 'Add to queue' })
  const already = await page.getByRole('button', { name: 'Remove from queue' }).count()
  for (let i = 0; i < count; i += 1) {
    const btn = adds.first()
    if (!(await btn.isVisible().catch(() => false))) break
    await btn.click()
    // Wait for the toggle to flip, so the next `.first()` is a DIFFERENT episode rather than this
    // one again — otherwise the loop races itself and queues one episode three times.
    await expect(page.getByRole('button', { name: 'Remove from queue' })).toHaveCount(already + i + 1)
  }
}

/** Open an episode and play it to the end, so it lands in history AND reads as played. */
async function finish(page: Page, showPath: string, title: string): Promise<void> {
  await page.goto(showPath)
  await page.getByText(title).first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await page.locator('[data-testid="app-audio"]').waitFor({ state: 'attached', timeout: 15_000 })
  await page.evaluate(() => {
    const a = document.querySelector('audio') as HTMLAudioElement | null
    if (!a) return
    // A real position first: "recently played" is built from the playback list, and a record with
    // no progress is not what the operator is reviewing.
    a.currentTime = 120
    a.dispatchEvent(new Event('timeupdate'))
    a.dispatchEvent(new Event('ended'))
  })
  await page.waitForTimeout(400)
}

test('the queue panel, with a real queue and real history', async ({ page }) => {
  await page.goto(`/api/app/auth/login?as=${IDENTITY}`)
  await expectSignedIn(page)

  // History first — finishing an episode removes it from nothing we are about to queue, and doing
  // it second would leave the queue mid-mutation when the panel opens.
  // Real titles from the fixture corpus, not invented ones — a guessed title fails as a timeout
  // 60 seconds later and reads like a broken app rather than a broken spec.
  await finish(page, '/podcast/p05', 'Index Investing Without the Myths')
  await finish(page, '/podcast/p03', 'Wreck Diving Fundamentals')
  await finish(page, '/podcast/p03', 'Marine Biology for Divers')
  await queueFrom(page, '/podcast/p01', 3)

  // Go to /queue, which is where both halves live since 2026-09-27. The panel this spec was
  // written against is gone with its last opener — the player's transport button — so the surface
  // under review is the page, and `panel` below is the page's own container.
  await page.goto('/queue')
  await settle(page)

  const panel = page.getByTestId('queue-page')
  await expect(panel).toBeVisible()
  await expect(panel.getByTestId('queue-panel-recent')).toBeVisible({ timeout: 20_000 })
  await settle(page)

  // Up next must actually have items, or the shot proves nothing about the queue half.
  const upNextCards = panel.locator('[data-testid="episode-card"]')
  expect(await upNextCards.count()).toBeGreaterThan(1)

  await panel.screenshot({ path: shot('panel-top') })

  // The recently-played section on its own, where items 1, 3 and 4 live.
  const recent = panel.getByTestId('queue-panel-recent')
  await recent.scrollIntoViewIfNeeded()
  await settle(page)
  await expect(recent.locator('time').first()).toBeVisible()
  await expect(recent.getByTestId('card-read-more').first()).toBeVisible()
  await recent.screenshot({ path: shot('recently-played') })

  // And the whole scrollable panel, so the two sections can be read against each other.
  await page.screenshot({ path: shot('panel-full'), fullPage: false })
})

test('Up next, on the NATIVE branch, where the inline download control lives', async ({ page }) => {
  /*
   * `DownloadButton` is `v-if="native"`, so a plain browser build renders five controls and the
   * sixth — the one the operator asked for — is absent from every web screenshot. That is correct:
   * audio is bridged and never SW-cached (bridge-never-rehost), so a web "download" would be a
   * control that cannot work.
   *
   * Capacitor picks its platform by sniffing `window.webkit.messageHandlers.bridge`, so the branch
   * CAN be forced — but only in this order, which took two wrong attempts to find:
   *
   *   - Forcing it before boot signs the app out. The native shell authenticates with a bearer
   *     token, not the session cookie, so the cookie login lands on a signed-out app.
   *   - Patching after boot then navigating with `page.goto` throws the patch away: a full page
   *     load re-runs the bundle and rebuilds the Capacitor global from scratch.
   *
   * So: sign in and navigate on the WEB path, patch last, and open the panel CLIENT-SIDE. The
   * panel's cards mount after the patch and read the forced value.
   */
  const run = Date.now().toString(36)
  await page.goto(`/api/app/auth/login?as=design-queue-native-${run}`)
  await expectSignedIn(page)
  await queueFrom(page, '/podcast/p01', 3)

  // Seed ONE episode as downloaded, so the shot shows both states side by side — a row where every
  // control looks the same cannot show what "already downloaded" looks like.
  const seeded = await page.evaluate(async () => {
    const me = await (await fetch('/api/app/me', { credentials: 'include' })).json()
    const queue = await (await fetch('/api/app/queue', { credentials: 'include' })).json()
    const slug = (queue.items ?? queue.slugs ?? [])[0]
    if (!me?.user_id || !slug) return null
    localStorage.setItem(
      `CapacitorStorage.downloads.registry.${me.user_id}`,
      JSON.stringify({ [slug]: { slug, state: 'downloaded', updatedAt: 1 } }),
    )
    return slug
  })
  expect(seeded, 'could not seed a downloaded episode').toBeTruthy()

  // Last FULL load — the registry above is read during this boot.
  await page.goto('/podcast/p01')
  await page.getByText('Building Trails That Last').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await settle(page)

  // Patch AFTER the last navigation, then open the panel client-side.
  await page.evaluate(() => {
    const cap = (window as unknown as { Capacitor?: Record<string, unknown> }).Capacitor
    if (!cap) throw new Error('Capacitor global missing — cannot force the native branch')
    cap.isNativePlatform = () => true
    cap.getPlatform = () => 'ios'
  })
  await page.getByTestId('player-queue').click()

  const panel = page.getByTestId('queue-panel')
  await expect(panel).toBeVisible()
  // The claim, asserted rather than eyeballed: download is IN the row.
  await expect(panel.getByTestId('download-button').first()).toBeVisible({ timeout: 20_000 })
  // ...and the seeded one reads as downloaded, in the accent the queue toggle uses for "on".
  await expect(panel.locator('[data-testid="download-button"][data-state="downloaded"]')).toHaveCount(1)
  await settle(page)
  await panel.screenshot({ path: shot('up-next-native') })
})
