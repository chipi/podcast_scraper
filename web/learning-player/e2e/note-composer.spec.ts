import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * NoteComposer (NT.1/NT.2) — the one reusable "add a note" affordance for any note target. REAL API
 * over the committed corpus, NO mocks. Drives it on the SHOW page (`target="show"`), the surface
 * `capture.spec.ts` does not touch — that one covers notes on a highlight. Add → it lists with a
 * timestamp; delete → it is gone. Dictation (Web Speech) is not driven here: the mic renders only
 * behind the opt-in Settings flag AND a platform SpeechRecognition, and no real speech can be fed in
 * headless — it is unit-tested in `NoteComposer.test.ts`.
 */
test('add and delete a note on the show page', async ({ page }, testInfo) => {
  await signInIsolated(page, 'note-composer', testInfo)
  await page.goto('/podcast/p05')

  const composer = page.getByTestId('note-composer')
  await expect(composer).toBeVisible()

  const text = `e2e show note ${Date.now()}`
  await composer.getByTestId('note-input').fill(text)
  await composer.getByTestId('note-save').click()

  const item = composer.getByTestId('note-item').filter({ hasText: text })
  await expect(item).toBeVisible()

  await item.getByTestId('note-delete').click()
  await expect(composer.getByTestId('note-item').filter({ hasText: text })).toHaveCount(0)
})
