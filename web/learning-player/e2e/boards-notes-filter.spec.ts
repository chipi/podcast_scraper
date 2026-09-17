import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

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
  await page.getByRole('tab', { name: 'Boards' }).click()
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

  // The strip offers a chip per kind PRESENT, and nothing for kinds with no notes.
  const strip = page.getByTestId('notes-type-filter')
  await expect(strip).toBeVisible()
  await expect(page.getByTestId('notes-type-show')).toBeVisible()
  await expect(page.getByTestId('notes-type-storyline')).toBeVisible()
  await expect(page.getByTestId('notes-type-person')).toHaveCount(0)

  // Selecting a kind shows only that kind.
  await page.getByTestId('notes-type-storyline').click()
  await expect(
    notes.getByTestId('collections-note').filter({ hasText: storylineText }),
  ).toBeVisible()
  await expect(notes.getByTestId('collections-note').filter({ hasText: showText })).toHaveCount(0)

  // "All" clears it in one tap.
  await page.getByTestId('notes-type-all').click()
  await expect(notes.getByTestId('collections-note').filter({ hasText: showText })).toBeVisible()

  // A filter that matches nothing must not delete the control that clears it: the strip survives
  // and the section says so instead of rendering a blank that reads as a bug.
  await page.getByTestId('notes-type-storyline').click()
  await page.getByTestId('collections-search').fill('zzzz-matches-nothing')
  await expect(page.getByTestId('collections-notes-empty')).toBeVisible()
  await expect(strip).toBeVisible()
  await expect(page.getByTestId('notes-type-all')).toBeVisible()

  // Leave no trace.
  await page.getByTestId('collections-search').fill('')
  await page.getByTestId('notes-type-all').click()
  for (const text of [showText, storylineText]) {
    const row = notes.getByTestId('collections-note').filter({ hasText: text })
    await row.getByRole('button', { name: 'Remove' }).click()
    await expect(row).toHaveCount(0)
  }
})
