import { expect, test, type Locator, type Page } from '@playwright/test'
import { signInIsolated, tapAndRecordTop } from './helpers'

/**
 * You land where you meant to (operator 2026-10-04). REAL API, no mocks, phone-sized so every
 * section that matters sits below the fold — at desktop height most of them do not, and a scroll
 * assertion that starts at 0 proves nothing.
 *
 * 1. A stack opened from an episode — topic → theme → person, each ON TOP of the last — closes back
 *    one layer at a time, and every layer is still scrolled to the spot it was opened from.
 * 2. A note's "Open" in the Library lands on that page's notes section, not its top; an episode's
 *    note opens the episode-notes panel scrolled to the notes.
 */

/** scrollTop of the nearest scrolling ancestor — the panel body or a sheet, not the page. */
function scrollerTop(el: Locator): Promise<number> {
  return el.evaluate((node) => {
    let n: HTMLElement | null = node.parentElement
    while (n) {
      const o = getComputedStyle(n).overflowY
      if ((o === 'auto' || o === 'scroll') && n.scrollHeight > n.clientHeight) return n.scrollTop
      n = n.parentElement
    }
    return document.scrollingElement?.scrollTop ?? 0
  })
}

/** Where a control is, both as its scroller's offset and as its position on screen. */
interface Place {
  scroll: number
  screen: number
}

async function place(el: Locator): Promise<Place> {
  // Measure where the reader will tap it: ON screen and no longer moving. A card still loading
  // shifts what is under the finger, and a baseline read mid-shift (902px, below the fold of a
  // 760px screen) is not where the reader saw it — no restore can be held to that.
  const viewport = page_(el).viewportSize()!.height
  await expect
    .poll(
      async () => {
        await el.scrollIntoViewIfNeeded()
        const a = (await el.boundingBox())!.y
        await el.page().waitForTimeout(300)
        const b = (await el.boundingBox())!
        return Math.abs(a - b.y) < 1 && b.y >= 0 && b.y + b.height <= viewport
      },
      { intervals: [0], timeout: 15_000, message: 'the control never settled on screen' },
    )
    .toBe(true)
  return { scroll: await scrollerTop(el), screen: (await el.boundingBox())!.y }
}

const page_ = (el: Locator): Page => el.page()

/** Settle `el` on screen, then tap it; `screen` is where it was AT the tap (helpers). */
async function tap(el: Locator): Promise<Place> {
  const settled = await place(el)
  return { scroll: settled.scroll, screen: await tapAndRecordTop(el) }
}

async function expectBackAt(el: Locator, before: Place, what: string): Promise<void> {
  // The contract: the control the reader tapped is back where they SAW it — fully in view, at the
  // same spot on screen. Not a scroll offset: content around it can finish loading after the
  // restore, and the offset then has to change to keep the control where it was.
  await expect(el, `${what}: the control it was opened from is not back in view`).toBeInViewport({
    ratio: 1,
  })
  await expect
    .poll(async () => Math.round(Math.abs((await el.boundingBox())!.y - before.screen)), {
      message: `${what}: not at the spot on screen it was tapped at (${Math.round(before.screen)}px)`,
    })
    .toBeLessThan(12)
}

/** The ✕ of the sheet on top. `.last()` in DOM order can be a card UNDER it (sheets teleport). */
const topSheetClose = (page: Page) => page.locator('[data-testid="ec-dismiss"][aria-label="Close"]')

async function openRiskPanelEpisode(page: Page): Promise<void> {
  await page.goto('/podcast/p05')
  await page.getByText('The Risk Panel: Diversify or Concentrate?').first().click()
  await expect(page).toHaveURL(/\/episode\//)
}

test.use({ viewport: { width: 390, height: 760 } })

/**
 * The first chip (from the bottom, so it is below the fold) whose card contains `inside`. Not every
 * person has related people and not every topic has similar topics, so walk until one does. Leaves
 * that card OPEN and returns the chip with the panel offset it was tapped at.
 */
async function openChipWith(
  page: Page,
  chipTestId: string,
  inside: string,
): Promise<{ chip: Locator; top: Place }> {
  const chips = page.getByTestId(chipTestId)
  await expect(chips.first()).toBeVisible()
  for (let i = (await chips.count()) - 1; i >= 0; i--) {
    const chip = chips.nth(i)
    await chip.scrollIntoViewIfNeeded()
    const top = await tap(chip)
    // The card's sections render only after its fetch; its last control renders with them. Counting
    // before that read every card as empty — the flake measured 2026-10-04 (failed, then passed on
    // retry) and the chain test's "no topic belongs to a theme" on a corpus where five do.
    await expect(page.getByTestId('ec-search-library').first()).toBeAttached()
    if ((await page.getByTestId(inside).count()) > 0) return { chip, top }
    await page.getByTestId('ec-dismiss').first().click()
    await expect(chip).toBeVisible()
  }
  throw new Error(`no ${chipTestId} in this episode opens a card with ${inside}`)
}


test('episode → topic → theme → person: each Back lands on the exact spot it was opened from', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'back-stack-chain', testInfo)
  await openRiskPanelEpisode(page)
  await page.getByTestId('player-open-insights').click()

  // A topic whose card names a theme. Not every topic has one, so walk the chips until one does.
  const { chip, top: chipTop } = await openChipWith(page, 'kp-topic-chip', 'ec-theme-link')
  expect(chipTop.scroll, 'the topic chips are not below the fold, so this proves nothing').toBeGreaterThan(100)

  // Topic card → its theme, opened ON TOP.
  const themeLink = page.getByTestId('ec-theme-link')
  await themeLink.scrollIntoViewIfNeeded()
  const themeLinkTop = await tap(themeLink)
  const themeSheet = page.getByTestId('theme-view')
  await expect(themeSheet).toBeVisible()

  // Theme → a person from its Top voices, ON TOP again.
  const voice = themeSheet.getByTestId('ec-top-voice').last()
  await expect(voice, 'the theme has no Top voices to open a person from').toBeAttached()
  await voice.scrollIntoViewIfNeeded()
  const voiceTop = await tap(voice)
  await expect(topSheetClose(page)).toBeVisible()

  // Back, one layer at a time.
  await topSheetClose(page).click() // person
  await expectBackAt(voice, voiceTop, 'theme after closing the person')
  await themeSheet.getByTestId('theme-card-close').click() // theme
  await expectBackAt(themeLink, themeLinkTop, 'topic card after closing the theme')
  await page.getByTestId('ec-dismiss').first().click() // topic
  await expectBackAt(chip, chipTop, 'episode notes after closing the topic')
})

test('every way out of the episode notes comes back to the spot it left from', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'back-every-combination', testInfo)
  await openRiskPanelEpisode(page)
  await page.getByTestId('player-open-insights').click()
  const back = () => page.getByTestId('ec-dismiss').first()

  // episode → person, then person → related person (the card's own stack) and back twice.
  const person = await openChipWith(page, 'kp-person-chip', 'ec-related-person')
  expect(person.top.scroll, 'the person chips are not below the fold, so this proves nothing').toBeGreaterThan(100)
  const related = page.getByTestId('ec-related-person').last()
  await related.scrollIntoViewIfNeeded()
  const relatedTop = await tap(related)
  // The card body is the SAME scroller for the next entity; it must start at the new one's top.
  const cardBody = page.getByTestId('ec-search-library').first()
  await expect(cardBody).toBeAttached()
  await expect.poll(() => scrollerTop(cardBody), { message: 'the related person opened mid-way down' }).toBeLessThan(8)
  await back().click()
  await expectBackAt(related, relatedTop, 'person after Back from a related person')
  await back().click()
  await expectBackAt(person.chip, person.top, 'episode notes after Back from the person')

  // episode → topic, then topic → similar topic (stack) and topic → person (a sheet ON TOP).
  const topic = await openChipWith(page, 'kp-topic-chip', 'ec-similar-topic')
  const similar = page.getByTestId('ec-similar-topic').last()
  await similar.scrollIntoViewIfNeeded()
  const similarTop = await tap(similar)
  await back().click()
  await expectBackAt(similar, similarTop, 'topic after Back from a similar topic')

  // Pinned to the element by its href: a page-wide `.last()` re-resolves on every use, and once a
  // sheet with its own voices is open it names a different element than the one tapped.
  const lastVoice = page.getByTestId('ec-top-voice').last()
  const voiceHref = (await lastVoice.count()) > 0 ? await lastVoice.getAttribute('aria-label') : null
  const voice = page.getByTestId('ec-top-voice').and(page.getByLabel(voiceHref ?? '', { exact: true })).first()
  if (voiceHref) {
    await voice.scrollIntoViewIfNeeded()
    const voiceTop = await tap(voice)
    await expect(topSheetClose(page)).toBeVisible() // the person sheet over the topic
    await topSheetClose(page).click()
    await expectBackAt(voice, voiceTop, 'topic after closing the person sheet on top of it')
  }
  await back().click()
  await expectBackAt(topic.chip, topic.top, 'episode notes after Back from the topic')
})

test("a note's Open lands on the notes — a page's section, an episode's panel", async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'note-open-lands-on-notes', testInfo)
  const tag = `${Date.now()}-${testInfo.repeatEachIndex}-${Math.random().toString(36).slice(2, 7)}`
  const pageNote = `e2e storyline landing ${tag}`
  const episodeNote = `e2e episode landing ${tag}`

  await page.goto('/storyline/topic:risk-management')
  const composer = page.getByTestId('note-composer')
  await composer.getByTestId('note-input').fill(pageNote)
  await composer.getByTestId('note-save').click()
  await expect(composer.getByTestId('note-item').filter({ hasText: pageNote })).toBeVisible()

  await openRiskPanelEpisode(page)
  await page.getByTestId('player-open-insights').click()
  const panelComposer = page.getByTestId('knowledge-panel').getByTestId('note-composer')
  await panelComposer.getByTestId('note-input').fill(episodeNote)
  await panelComposer.getByTestId('note-save').click()
  await expect(panelComposer.getByTestId('note-item').filter({ hasText: episodeNote })).toBeVisible()

  const openFromLibrary = async (text: string) => {
    await page.goto('/library')
    await page.getByRole('tab', { name: 'Boards' }).click()
    const row = page.getByTestId('collections-note').filter({ hasText: text })
    await row.getByRole('link', { name: 'Open' }).click()
  }

  await openFromLibrary(pageNote)
  await expect(page).toHaveURL(/\/storyline\/.*#notes$/)
  await expect(page.getByTestId('note-composer')).toBeInViewport()
  expect(await page.evaluate(() => window.scrollY), 'still at the top of the page').toBeGreaterThan(200)

  await openFromLibrary(episodeNote)
  await expect(page).toHaveURL(/[?&]notes=1/)
  const notes = page.getByTestId('knowledge-panel').getByTestId('note-composer')
  await expect(notes).toBeInViewport()
  expect(await scrollerTop(notes), 'the panel opened at its top').toBeGreaterThan(100)

  // Leave no trace.
  await page.goto('/library')
  await page.getByRole('tab', { name: 'Boards' }).click()
  for (const text of [pageNote, episodeNote]) {
    const row = page.getByTestId('collections-note').filter({ hasText: text })
    await row.getByRole('button', { name: 'Remove' }).click()
    await expect(row).toHaveCount(0)
  }
})
