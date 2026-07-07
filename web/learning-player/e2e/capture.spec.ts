import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

const serious = (vs: { impact?: string | null }[]) =>
  vs.filter((v) => v.impact === 'critical' || v.impact === 'serious')

/**
 * P2 Capture end-to-end — REAL API over the COMMITTED validation corpus, NO mocks. Drives the full
 * listen→capture→review loop: sign in (mock OAuth), open an episode, mark a moment + save a
 * transcript line, then review them in the Library "Highlights" tab, attach a note, and confirm the
 * Markdown export link. Per-user state is written to the gitignored APP_DATA_DIR.
 *
 * Idempotency: the two parallel projects (mobile + desktop) share one mock-user store, so we only
 * ever ADD — the moment capture is monotonic, and the transcript-line save is guarded on its
 * pressed state (never toggled off). Assertions are visibility-based, never exact counts.
 *
 * Episode: the newest, "Index Investing Without the Myths" ("Long Horizon Notes" / fixture p05).
 */
test('sign in → mark a moment + save a line → review in Library Highlights + add a note', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'capture', testInfo)

  // Open the episode and wait for its transcript (the real metadata → /segments path).
  await page.goto('/')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page.getByText(/Index funds are not a strategy/).first()).toBeVisible()

  // a11y: the signed-in player (with the auth-gated capture controls present) has no serious axe
  // violations — the capture affordances are keyboard- and SR-reachable.
  const playerAxe = await new AxeBuilder({ page }).analyze()
  expect(serious(playerAxe.violations)).toEqual([])

  // Mark a moment (auth-gated hero control) — monotonic add.
  await page.getByRole('button', { name: 'Mark this moment' }).click()

  // Save a transcript line as a span highlight (idempotent: only if not already pressed).
  const lineSave = page
    .getByRole('button', { name: 'Save highlight — your selected text, or this whole line' })
    .first()
  await expect(lineSave).toBeVisible()
  if ((await lineSave.getAttribute('aria-pressed')) === 'false') {
    await lineSave.click()
  }
  await expect(page.getByRole('button', { name: 'Line saved — tap to remove' }).first()).toBeVisible()

  // Review in Library → Highlights tab (auth-gated), served by the real API.
  await page.goto('/library')
  await page.getByRole('button', { name: 'Highlights' }).click()

  // The episode group renders with its captured items (the marked moment is always present).
  await expect(page.getByText('Marked moment').first()).toBeVisible()

  // a11y: the Highlights review surface (swatch pickers, colour filter, notes, export) is clean.
  const highlightsAxe = await new AxeBuilder({ page }).analyze()
  expect(serious(highlightsAxe.violations)).toEqual([])

  // The Markdown export link points at the real export route.
  const exportLink = page.getByRole('link', { name: 'Export Markdown' })
  await expect(exportLink).toHaveAttribute('href', /\/api\/app\/highlights\/export\.md/)

  // Attach a note to the first highlight and confirm it persists in the view.
  const noteText = `e2e note ${Date.now()}`
  await page.getByRole('button', { name: 'Add note' }).first().click()
  await page.getByRole('textbox', { name: 'Note' }).first().fill(noteText)
  await page.getByRole('button', { name: 'Save', exact: true }).first().click()
  await expect(page.getByText(noteText)).toBeVisible()
})
