import { expect, test, type Locator, type Page } from '@playwright/test'
import { signInIsolated, tapAndRecordTop, showEveryonesTrends } from './helpers'

/**
 * StorylineView (F4.5), reached from the "Storylines" tab of Discover's Trends. REAL API over the
 * committed corpus, NO mocks. On Discover a tapped storyline opens its PAGE (`/storyline/<anchor>`);
 * Home opened it as the StorylineCard overlay until Trends left Home (operator 2026-10-07). The
 * overlay is still reached from a topic card's storyline link — `entity-and-rails-invariants.spec.ts`
 * and `entity-pages.spec.ts` drive it from there.
 */
/**
 * The first storyline row that is ACTUALLY OPENABLE.
 *
 * `.first()` was the bug behind two years of "flake" in this file. `DiscoveryList` deliberately
 * renders a storyline whose anchor topic the server could not resolve as INERT — its inner button
 * carries `aria-disabled="true"` and the tap does nothing, by design ("a row whose anchor the
 * server could not resolve is genuinely not openable, and pretending otherwise is what produced a
 * dead tap"). Rows are ordered by momentum, so WHICH one sorts first changes between runs. Clicking
 * the first row therefore opened the card on some runs and did nothing on others — measured 2026-09-25
 * as 2 failures in 3 runs, serial, on an idle machine, and identically on the pre-refactor tree.
 *
 * That is not timing and no wait fixes it. An earlier attempt added one (see the comment below in
 * the follow test) and left the real cause in place.
 *
 * Fails LOUDLY when nothing is openable rather than skipping: the app itself renders a
 * `discovery-all-inert` notice for that state, so it is a real condition worth failing on, not an
 * excuse to assert nothing.
 */
async function firstOpenableStorylineRow(page: Page): Promise<Locator> {
  // SCOPED to the storyline list. Every kind-tab renders rows under the SAME `discovery-row`
  // testid, so an unscoped lookup also matches the PREVIOUS tab's topic rows, which linger in the
  // DOM until the re-render lands. That is why the row count moved between 1 and 4 across
  // otherwise identical runs, and why `.first()` sometimes resolved to a topic row that opens a
  // topic card instead of a storyline (2026-09-25). The file already carried a comment describing
  // this leak; the fix at the time added a wait, which does not scope anything.
  const list = page.getByTestId('discovery-list-storyline')
  await expect(list).toBeVisible()
  const rows = list.getByTestId('discovery-row')
  await expect(rows.first()).toBeVisible()
  const total = await rows.count()
  for (let i = 0; i < total; i++) {
    const row = rows.nth(i)
    if ((await row.locator('[aria-disabled="true"]').count()) === 0) return row
  }
  throw new Error(
    `all ${total} storyline rows are inert (no resolvable anchor topic), so none can be opened. ` +
      `The corpus or the trending endpoint is not returning anchor ids — this is the state the ` +
      `app flags with "discovery-all-inert".`
  )
}

test('a Discover storyline row opens the storyline page — members and episodes, not a shell', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'storyline', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)

  await page.getByTestId('discovery-tab-storyline').click()
  // The first OPENABLE row, not `.first()` — see `firstOpenableStorylineRow`.
  const row = await firstOpenableStorylineRow(page)
  await row.click()

  await expect(page).toHaveURL(/\/storyline\//)
  const view = page.getByTestId('storyline-view')
  await expect(view).toBeVisible()

  // F2.2: a storyline is favoritable (the shared heart), distinct from Follow. Toggling it flips
  // the pressed state — it lands in Library › Saved like any other kind.
  const heart = view.locator('.lp-fav').first()
  await expect(heart).toBeVisible()
  const before = await heart.getAttribute('aria-pressed')
  await heart.click()
  await expect(heart).not.toHaveAttribute('aria-pressed', before ?? 'false')

  // Not an empty shell: it names the storyline (h1) and lists its member topics.
  await expect(view.locator('h1')).not.toHaveText('...')
  await expect(view.getByText("Couldn't load the topics in this storyline.")).toHaveCount(0)
  await expect(view.getByRole('listitem').first()).toBeVisible()
})

test('a storyline opened from Discover can be followed', async ({ page }, testInfo) => {
  // The committed corpus trends exactly one storyline (thc:managing-risk), and it carries a theme
  // cluster, so the first openable row is the one that offers Follow. If that ever stops being
  // true the failure names the row, rather than the old open-then-skip that reported "1 skipped"
  // for a run that had checked nothing (2026-09-25).
  await signInIsolated(page, 'storyline-follow', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  await page.getByTestId('discovery-tab-storyline').click()
  const row = await firstOpenableStorylineRow(page)
  const label = (await row.innerText()).split('\n')[0]
  await row.click()
  await expect(page).toHaveURL(/\/storyline\//)
  await expect(page.getByTestId('storyline-view')).toBeVisible()

  // Follow renders a network round-trip after the view (StorylineView resolves the cluster id from
  // the anchor topic first) — `toBeVisible` waits; `isVisible` would not.
  const follow = page.getByTestId('storyline-follow')
  await expect(follow, `the storyline "${label}" offered no follow control`).toBeVisible({ timeout: 10_000 })
  const before = await follow.getAttribute('aria-pressed')
  await follow.click()
  await expect(follow).not.toHaveAttribute('aria-pressed', before ?? 'false')
})

test('the storyline PAGE shows its people as Top voices — the topic card\'s grid, not chips', async ({
  page,
}, testInfo) => {
  // Operator 2026-09-30: the topic card drew its people as an avatar grid ("Top voices") while the
  // storyline page listed the same people as "Related people" chips. Both now render TopVoices.
  // The people section is PAGE-only (the overlay stays a compact preview); Discover opens the page.
  await signInIsolated(page, 'storyline-voices', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  await page.getByTestId('discovery-tab-storyline').click()
  const row = await firstOpenableStorylineRow(page)
  await row.click()
  await expect(page).toHaveURL(/\/storyline\//)
  const view = page.getByTestId('storyline-view')
  await expect(view).toBeVisible()
  const voices = view.getByTestId('ec-top-voices')
  await expect(voices).toBeVisible()
  await expect(voices.getByText('Top voices')).toBeVisible()
  await expect(voices.getByTestId('ec-top-voice').first()).toBeVisible()
  expect(await voices.getByTestId('ec-top-voice').count()).toBeLessThanOrEqual(8)
  // On a page each voice is a real link to the person.
  await expect(voices.getByTestId('ec-top-voice').first()).toHaveAttribute('href', /\/person\//)
  await expect(view.getByText('Related people')).toHaveCount(0)
})

test('Back from a person opened in Top voices returns to Top voices, not the top of the page', async ({
  page,
}, testInfo) => {
  // Operator 2026-10-04: Back landed at the top of the page the person was opened from, so the
  // reader had to find their place again. Top voices sits beside the topics, inside the first screen
  // on a phone, so the reader scrolls it up near the top first: Back must restore that offset.
  await page.setViewportSize({ width: 390, height: 760 })
  await signInIsolated(page, 'storyline-back-scroll', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  await page.getByTestId('discovery-tab-storyline').click()
  const row = await firstOpenableStorylineRow(page)
  await row.click()
  await expect(page).toHaveURL(/\/storyline\//)
  const voice = page.getByTestId('storyline-view').getByTestId('ec-top-voice').first()
  await voice.scrollIntoViewIfNeeded()
  const voiceTop = (await voice.boundingBox())!.y
  await page.evaluate((y) => window.scrollBy(0, y), voiceTop - 120)
  const before = await page.evaluate(() => window.scrollY)
  expect(before, 'the page did not scroll, so this proves nothing').toBeGreaterThan(200)

  const seenAt = await tapAndRecordTop(voice)
  await expect(page).toHaveURL(/\/person\//)
  await page.getByTestId('ec-dismiss').click() // the person page's ✕ — a history Back
  await expect(page).toHaveURL(/\/storyline\//)
  // The voice tapped is back on screen, whole — not an exact offset: rails above it can finish
  // loading after the restore, and scroll anchoring then shifts the offset to keep it in view.
  await expect(voice).toBeInViewport({ ratio: 1 })
  await expect
    .poll(async () => Math.round(Math.abs((await voice.boundingBox())!.y - seenAt)), {
      message: 'the voice is not back at the spot on screen it was tapped at',
    })
    .toBeLessThan(12)
})
