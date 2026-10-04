import { expect, test, type Locator, type Page } from '@playwright/test'
import { signInIsolated, tapAndRecordTop } from './helpers'

/**
 * StorylineView (F4.5) — the storyline overlay (StorylineCard), reached from the Home
 * "Storylines" discovery tab. REAL API over the committed corpus, NO mocks. Tapping a storyline
 * row on Home opens the StorylineCard overlay on top (via `?storyline=` history entry).
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

test('Home storyline row opens the storyline overlay — members and episodes, not a shell', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'storyline', testInfo)
  await page.goto('/')

  // Storylines is one of the Home discovery kind-tabs.
  await page.getByTestId('discovery-tab-storyline').click()
  // The first OPENABLE row, not `.first()` — see `firstOpenableStorylineRow`.
  const row = await firstOpenableStorylineRow(page)
  await row.click()

  // Clicking a storyline row opens the StorylineCard overlay on top (with ?storyline= in the URL).
  const card = page.getByTestId('storyline-card')
  await expect(card).toBeVisible()
  await expect(page).toHaveURL(/[?&]storyline=/)
  const view = card.getByTestId('storyline-view')
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

test('the storyline overlay can be followed, when it carries a theme cluster', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'storyline-follow', testInfo)
  await page.goto('/')
  await page.getByTestId('discovery-tab-storyline').click()

  // SEARCH for a storyline that carries the affordance, rather than opening one and skipping when
  // it does not (2026-09-25).
  //
  // The old shape was: open `.first()`, and if no follow control appeared, `test.skip()`. Two
  // things were wrong with that, and together they made this test worthless:
  //
  //   1. `.first()` is not a stable choice — rows sort by momentum and an inert row opens nothing.
  //      An earlier fix added a wait for the list, which addressed a DIFFERENT race and left this
  //      one in place; the comment it left behind made the remaining failures look already-handled.
  //   2. A silent skip reports the same colour as a pass. Whether this behaviour was verified or
  //      quietly abandoned was invisible in the summary line, so nobody could tell that a run which
  //      said "1 skipped" had checked nothing at all.
  //
  // Follow renders only for a storyline whose card resolves a `thc:` cluster id, which is a
  // property of the DATA, not of timing. So walk the openable rows until one offers it. If none
  // does, FAIL and say how many were tried — a corpus that cannot exercise storyline-follow is a
  // corpus defect worth failing on, not a reason to assert nothing.
  // Scoped to the storyline list for the same reason as the helper above — the tabs share the
  // `discovery-row` testid and the previous tab's rows outlive the tap.
  const storylineList = page.getByTestId('discovery-list-storyline')
  await expect(storylineList).toBeVisible()
  const rows = storylineList.getByTestId('discovery-row')
  // WAIT for a row before counting. `locator.count()` does NOT auto-wait — it snapshots whatever
  // is in the DOM at that instant. The list CONTAINER becomes visible before its rows render, so
  // counting here returned 0, the loop never ran, and the failure read "opened 0 of 0 storyline
  // rows and none carried a follow control" — which blamed the corpus for a test that had not
  // looked at it (2026-09-25). Every other auto-waiting assertion in Playwright hides this, which
  // is what makes `count()` worth a comment.
  await expect(rows.first()).toBeVisible()
  const total = await rows.count()
  let follow = page.getByTestId('storyline-follow')
  let opened = 0
  for (let i = 0; i < total; i++) {
    const row = rows.nth(i)
    if ((await row.locator('[aria-disabled="true"]').count()) > 0) continue
    await row.click()
    const card = page.getByTestId('storyline-card')
    await expect(card).toBeVisible()
    await expect(card.getByTestId('storyline-view')).toBeVisible()
    opened += 1
    follow = page.getByTestId('storyline-follow')
    // WAIT for it. `isVisible()` is an INSTANT check — like `count()`, it does not auto-wait, and
    // almost every other Playwright call does, which is what makes these two worth calling out.
    // `StorylineView` resolves the cluster id from `getTopicCard(anchorTopicId)` BEFORE it can
    // render follow (`v-if="auth.isAuthenticated && storylineId"`), so the control appears a
    // network round-trip after the card does. Checking instantly always answered "no", the loop
    // pressed Escape before the control existed, and the failure blamed the corpus for data the
    // API had in fact returned — verified directly: `/topics/topic:risk-management` returns
    // `storyline_id = thc:managing-risk` (2026-09-25).
    if (await follow.waitFor({ state: 'visible', timeout: 10_000 }).then(() => true, () => false)) {
      break
    }
    // Not this one — close the overlay and try the next openable row.
    await page.keyboard.press('Escape')
    await expect(card).toBeHidden()
    follow = page.getByTestId('storyline-follow')
  }
  // NAME THE ROWS. "0 of 4" says four rows were unopenable but not WHAT they were, and the answer
  // decides the fix: storyline labels mean the anchors did not resolve, topic labels mean the
  // previous tab's rows are still rendering under the storyline container.
  const labels = await rows.allInnerTexts()
  const containers = await page.getByTestId('discovery-list-storyline').count()
  const topicLists = await page.getByTestId('discovery-list-topic').count()
  expect(
    await follow.waitFor({ state: 'visible', timeout: 10_000 }).then(() => true, () => false),
    `opened ${opened} of ${total} storyline rows and none carried a follow control.\n` +
      `Rows on screen: ${labels.map((l) => JSON.stringify(l.split('\n')[0])).join(', ')}\n` +
      `storyline containers: ${containers}, topic containers still mounted: ${topicLists}\n` +
      `The API returns exactly one storyline (thc:managing-risk) for this corpus, so anything else ` +
      `here is the wrong list under the right container.`
  ).toBe(true)

  const before = await follow.getAttribute('aria-pressed')
  await follow.click()
  await expect(follow).not.toHaveAttribute('aria-pressed', before ?? 'false')
})

test('the storyline PAGE shows its people as Top voices — the topic card\'s grid, not chips', async ({
  page,
}, testInfo) => {
  // Operator 2026-09-30: the topic card drew its people as an avatar grid ("Top voices") while the
  // storyline page listed the same people as "Related people" chips. Both now render TopVoices.
  // The people section is PAGE-only (the overlay stays a compact preview), so open the storyline
  // from Home, take its anchor from `?storyline=`, and load the page itself.
  await signInIsolated(page, 'storyline-voices', testInfo)
  await page.goto('/')
  await page.getByTestId('discovery-tab-storyline').click()
  const row = await firstOpenableStorylineRow(page)
  await row.click()
  await expect(page).toHaveURL(/[?&]storyline=/)
  const anchor = new URL(page.url()).searchParams.get('storyline')
  expect(anchor, 'no ?storyline= anchor in the URL').toBeTruthy()

  await page.goto(`/storyline/${encodeURIComponent(anchor!)}`)
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
  // reader had to find their place again. Phone-sized, so Top voices sits below the fold.
  await page.setViewportSize({ width: 390, height: 760 })
  await signInIsolated(page, 'storyline-back-scroll', testInfo)
  await page.goto('/')
  await page.getByTestId('discovery-tab-storyline').click()
  const row = await firstOpenableStorylineRow(page)
  await row.click()
  await expect(page).toHaveURL(/[?&]storyline=/)
  const anchor = new URL(page.url()).searchParams.get('storyline')
  expect(anchor, 'no ?storyline= anchor in the URL').toBeTruthy()

  await page.goto(`/storyline/${encodeURIComponent(anchor!)}`)
  const voice = page.getByTestId('storyline-view').getByTestId('ec-top-voice').first()
  await voice.scrollIntoViewIfNeeded()
  const before = await page.evaluate(() => window.scrollY)
  expect(before, 'Top voices is not below the fold, so this proves nothing').toBeGreaterThan(200)

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
