import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * The surfaces this arc added to the END-USER app, in a real browser (#1914/#1925).
 *
 * They all had unit tests against mocked stores and NO browser coverage: the recap panel, the Home
 * prompt that points at it, the `?t=` deep link, and the Profile tile that was REMOVED because it
 * showed a fabricated number. Unit tests cannot see any of those wired to a real API, and the arc
 * has already shown twice that a store contract can change while its view keeps rendering.
 *
 * Native-only surfaces (DownloadButton, DownloadedList, DeviceSettings) are deliberately absent:
 * they are behind `isNative()`, render nothing in a browser, and are covered by the device tier
 * (`make test-app-ios-journey`).
 */

const EPISODE = 'Index Investing Without the Myths'

/** Play far enough to record listening time — the recap needs a position DELTA, not a position. */
async function listenFor(page: import('@playwright/test').Page, seconds: number): Promise<void> {
  // The store CONSTRUCTS the element on load (#1587), so it does not exist until the player page
  // has actually armed playback. Waiting for the transport is the observable signal for that.
  await expect(
    page
      .getByRole('button', { name: 'Play', exact: true })
      .or(page.getByText(/couldn.*load the audio|Audio.*unavailable/i)),
  ).toBeVisible()
  await page.locator('audio').waitFor({ state: 'attached' })
  await page.evaluate(async (target) => {
    const el = document.querySelector('audio') as HTMLAudioElement | null
    if (!el) throw new Error('no audio element')
    el.currentTime = 0
    await el.play().catch(() => {})
    // Step the playhead the way real playback would, so each save is a small forward delta and
    // the server's clamp accepts it. One jump would be a SEEK and would accrue nothing.
    for (let t = 1; t <= target; t += 1) {
      el.currentTime = t
      el.dispatchEvent(new Event('timeupdate'))
      await new Promise((r) => setTimeout(r, 20))
    }
    el.dispatchEvent(new Event('pause'))
  }, seconds)
}

test('the recap panel replaces the fabricated Hours tile and states its coverage', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'recap-panel', testInfo)

  await page.goto('/podcast/p05')
  await page.getByText(EPISODE).first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await listenFor(page, 6)

  await page.goto('/profile')

  // The recap panel is present and names the window it is describing.
  const recap = page.getByRole('heading', { name: 'Your listening' })
  await expect(recap).toBeVisible()
  await expect(page.getByRole('button', { name: 'Week', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'This year', exact: true })).toBeVisible()

  // It ALWAYS says how much of the window it actually has while that is partial — recording
  // started recently, so a bare total would be a lie of omission.
  await expect(page.getByText(/Recorded \d+ of \d+ days/)).toBeVisible()

  // The tile that showed `sum(position_seconds)` — a lifetime furthest-position snapshot — is
  // gone, and the panel that remains is titled for what it actually measures.
  await expect(page.getByText('Your activity')).toBeVisible()
  await expect(page.getByText('Hours', { exact: true })).toHaveCount(0)
})

test('the Home prompt appears once there is something to look back on, and opens Profile', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'recap-prompt', testInfo)

  // Nothing listened to yet: the prompt must not render. A row reading "0h · 0 episodes" spends
  // space telling the user what they already know.
  await page.goto('/')
  await expect(page.getByText('Your week in listening')).toHaveCount(0)

  await page.goto('/podcast/p05')
  await page.getByText(EPISODE).first().click()
  await listenFor(page, 6)

  await page.goto('/')
  const prompt = page.getByText('Your week in listening')
  await expect(prompt).toBeVisible()
  await prompt.click()
  await expect(page).toHaveURL(/\/profile/)
})

test('a ?t= deep link opens the episode AT that moment, not at the resume point', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'deep-link-t', testInfo)

  // Establish a resume position well away from the one the link will name.
  await page.goto('/podcast/p05')
  await page.getByText(EPISODE).first().click()
  await expect(page).toHaveURL(/\/episode\//)
  const slug = new URL(page.url()).pathname.split('/').pop() as string
  await listenFor(page, 8)

  // Follow a link that names a moment — a recap's saved line, a shared quote, an MCP citation.
  await page.goto(`/episode/${slug}?t=42`)
  await expect(page).toHaveURL(/t=42/)
  await page.locator('audio').waitFor({ state: 'attached' })

  // The link wins over the remembered position for this load; opening at the resume point would
  // silently drop the only reason the link existed.
  await expect
    .poll(
      async () =>
        page.evaluate(() => (document.querySelector('audio') as HTMLAudioElement)?.currentTime ?? 0),
      { timeout: 15_000 },
    )
    .toBeGreaterThan(40)
})

test('a malformed ?t= still opens the episode rather than breaking it', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'deep-link-bad-t', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText(EPISODE).first().click()
  // Await the navigation before reading the slug — otherwise the URL is still the SHOW page and
  // the "episode" link below points at /episode/p05, which renders nothing.
  await expect(page).toHaveURL(/\/episode\//)
  const slug = new URL(page.url()).pathname.split('/').pop() as string

  // Losing the moment is a shame; losing the episode is a broken link. `NaN` reaching
  // `currentTime` would throw, so it must be dropped rather than applied.
  await page.goto(`/episode/${slug}?t=not-a-number`)
  await expect(page.getByText(EPISODE).first()).toBeVisible()
  await expect(page.locator('audio')).toHaveCount(1)
})
