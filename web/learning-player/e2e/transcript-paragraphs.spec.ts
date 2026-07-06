import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Regression guard for the paragraph-transcript rewrite (this session). Segments are grouped into
 * flowing paragraphs (a speaker turn, broken at sentence boundaries every ~25s) with ONE timestamp
 * per paragraph — not one row per fragment. Real API over the committed corpus: fixture p05 carries
 * 62 undiarized segments over ~6 min, which collapse into far fewer, anchored paragraphs.
 *
 * If someone reverts to the old per-segment rows, the `p.leading-relaxed` grouping (and the
 * many-segments-per-paragraph assertion) breaks here.
 */
test('transcript renders as flowing paragraphs with per-paragraph capture', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'paragraphs', testInfo)
  await page.goto('/')
  await page.goto('/podcast/p05') // #1148: reach the episode via its show page (date-independent)
  await page.getByText('Index Investing Without the Myths').first().click()
  // The transcript renders from the real corpus (metadata → /segments path).
  await expect(page.getByText(/Index funds are not a strategy/).first()).toBeVisible()

  const paragraphs = page.locator('p.leading-relaxed')
  const segs = page.locator('[data-testid="seg"]')
  await expect(paragraphs.first()).toBeVisible()
  const pCount = await paragraphs.count()
  const segCount = await segs.count()

  // Density dropped: many segments collapse into far fewer paragraphs (one timestamp each).
  expect(segCount).toBeGreaterThan(10)
  expect(pCount).toBeGreaterThan(0)
  expect(pCount).toBeLessThan(segCount)

  // At least one paragraph flows MULTIPLE segments inline — the defining trait of the rewrite.
  const hasMultiSegParagraph = await paragraphs.evaluateAll((ps) =>
    ps.some((p) => p.querySelectorAll('[data-testid="seg"]').length >= 2),
  )
  expect(hasMultiSegParagraph).toBe(true)

  // The supporting quote renders INSIDE a flowing paragraph (the prose path, not a per-row div).
  await expect(
    page.locator('p.leading-relaxed', { hasText: 'Index funds are not a strategy' }).first(),
  ).toBeVisible()

  // Per-paragraph capture wiring: saving a paragraph flips its control to the saved state
  // (idempotent — guarded on aria-pressed so retries / the two parallel projects never toggle off).
  const save = page
    .getByRole('button', { name: 'Save highlight — your selected text, or this whole line' })
    .first()
  if ((await save.getAttribute('aria-pressed')) !== 'true') await save.click()
  await expect(
    page.getByRole('button', { name: 'Line saved — tap to remove' }).first(),
  ).toBeVisible()
})
