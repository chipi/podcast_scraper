import { expect, test } from '@playwright/test'
import { closeListSearch, listSearch, signInIsolated } from './helpers'

/**
 * Library → Boards, "Your notes": the kind filter (operator 2026-09-17). REAL API, no mocks.
 *
 * The chips filter by the ENTITY a note is attached to, and only kinds that actually have notes get
 * a chip — so the fixture has to create notes of two different kinds before the strip exists at all.
 * Both are made through the one shared `NoteComposer`, on the two pages that mount it at top level
 * (a show and a storyline), which is also a check that the same affordance really does serve
 * different targets.
 *
 * Notes are per-account in the gitignored APP_DATA_DIR, and this spec deletes what it creates so a
 * re-run meets the same account state it did the first time.
 */
test('notes on Boards filter by the entity kind they are attached to', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'boards-notes-filter', testInfo)

  const showText = `e2e show note ${Date.now()}`
  const storylineText = `e2e storyline note ${Date.now()}`

  // A note on a SHOW.
  await page.goto('/podcast/p09')
  const showComposer = page.getByTestId('note-composer')
  await expect(showComposer).toBeVisible()
  await showComposer.getByTestId('note-input').fill(showText)
  await showComposer.getByTestId('note-save').click()
  await expect(showComposer.getByTestId('note-item').filter({ hasText: showText })).toBeVisible()

  // A note on a STORYLINE. `topic:risk-management` is the ANCHOR topic id the storyline routes by —
  // not the `thc:` cluster id, which is label-derived and unstable.
  await page.goto('/storyline/topic:risk-management')
  const storylineComposer = page.getByTestId('note-composer')
  await expect(storylineComposer).toBeVisible()
  await storylineComposer.getByTestId('note-input').fill(storylineText)
  await storylineComposer.getByTestId('note-save').click()
  await expect(
    storylineComposer.getByTestId('note-item').filter({ hasText: storylineText }),
  ).toBeVisible()

  // Both notes now live in the Boards tab's notes section.
  await page.goto('/library')
  await page.getByRole('tab', { name: 'Collections' }).click()
  const notes = page.getByTestId('collections-notes')
  await expect(notes).toBeVisible()
  await expect(notes.getByTestId('collections-note').filter({ hasText: showText })).toBeVisible()
  await expect(
    notes.getByTestId('collections-note').filter({ hasText: storylineText }),
  ).toBeVisible()

  // Each row names its kind in words, not the raw enum — the same words as the chips.
  await expect(notes.getByTestId('collections-note').filter({ hasText: showText })).toContainText(
    /Show/i,
  )

  // ONE kind chip (operator 2026-10-15, it was a strip of pills): an option per kind PRESENT,
  // with its count, and nothing for kinds with no notes.
  const kind = page.getByTestId('notes-kind')
  const pick = async (value: string) => {
    await kind.click()
    await page.getByTestId(`notes-kind-opt-${value}`).click()
  }
  await kind.click()
  await expect(page.getByTestId('notes-kind-opt-show')).toBeVisible()
  await expect(page.getByTestId('notes-kind-opt-storyline')).toBeVisible()
  await expect(page.getByTestId('notes-kind-opt-person')).toHaveCount(0)
  await kind.click()

  // Selecting a kind shows only that kind.
  await pick('storyline')
  await expect(
    notes.getByTestId('collections-note').filter({ hasText: storylineText }),
  ).toBeVisible()
  await expect(notes.getByTestId('collections-note').filter({ hasText: showText })).toHaveCount(0)

  // "All notes" clears it.
  await pick('')
  await expect(notes.getByTestId('collections-note').filter({ hasText: showText })).toBeVisible()

  // A filter that matches nothing must not delete the control that clears it: the chip survives
  // and the section says so instead of rendering a blank that reads as a bug.
  await pick('storyline')
  const search = await listSearch(notes, 'notes-search')
  await search.fill('zzzz-matches-nothing')
  await expect(page.getByTestId('collections-notes-empty')).toBeVisible()
  await expect(kind).toHaveCount(1)

  // Leave no trace.
  await search.fill('')
  await closeListSearch(notes, 'notes-search')
  await pick('')
  for (const text of [showText, storylineText]) {
    const row = notes.getByTestId('collections-note').filter({ hasText: text })
    await row.getByRole('button', { name: 'Remove' }).click()
    await expect(row).toHaveCount(0)
  }
})

/**
 * "Open" on a note about an episode opens the episode-notes panel over the player, where the note is
 * shown — not the bare player (operator on device, 2026-10-09). Both an episode note and a note on a
 * highlight, in a real browser: the panel is a <dialog> that can only open once the episode loaded,
 * which is exactly the part a unit test cannot see.
 */
test("Open on an episode or highlight note shows the episode-notes panel", async ({ page }, testInfo) => {
  await signInIsolated(page, 'boards-notes-open', testInfo)
  const slug = ((await (await page.request.get('/api/app/episodes?page_size=1')).json()).items as Array<{ slug: string }>)[0].slug
  const run = Date.now().toString(36)
  const h = await page.request.post('/api/app/highlights', {
    data: { episode_slug: slug, kind: 'moment', start_ms: 61_000 },
  })
  expect(h.ok()).toBeTruthy()
  const highlightId = (await h.json()).id as string
  for (const [target, id, text] of [
    ['episode', slug, `Episode note ${run}`],
    ['highlight', highlightId, `Highlight note ${run}`],
  ] as const) {
    const r = await page.request.post('/api/app/notes', { data: { target, target_id: id, text } })
    expect(r.ok()).toBeTruthy()
  }

  for (const text of [`Episode note ${run}`, `Highlight note ${run}`]) {
    await page.goto('/library?tab=collections')
    const note = page.getByTestId('collections-note').filter({ hasText: text })
    await note.getByRole('link', { name: 'Open' }).click()
    await expect(page).toHaveURL(new RegExp(`/episode/${slug}\\?.*notes=1`))
    await expect(page.getByTestId('knowledge-panel'), `${text}: the notes panel stayed shut`).toBeVisible()
  }
})
