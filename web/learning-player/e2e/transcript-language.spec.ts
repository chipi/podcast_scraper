import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * S3.1 end to end: a listener can read a translated episode in English or in the original, and
 * find it by searching in EITHER language.
 *
 * p10 ("Sesiones de Sendero") is Spanish with a real English render captured from a DGX
 * translation run, so nothing here is synthetic: the Spanish body is what was spoken and the
 * English one is what the model produced.
 *
 * WHY THE SEARCH HALF IS IN THIS FILE. The control is only half the feature — switching the
 * transcript is useless if the episode is unfindable in the language it was spoken in (Goal 6).
 * Both halves depended on the same two defects (a reindex deleted the source-layer rows, and a
 * transcript-scoped search excluded the non-English tier), so they are asserted together: either
 * could regress without the other noticing.
 */

const SPANISH_LINE = /Bienvenidos de nuevo a Sesiones de Sendero/
const ENGLISH_LINE = /Welcome back to Trail Sessions/

async function openP10(page: import('@playwright/test').Page) {
  await page.goto('/podcast/p10')
  await page.getByText('Construyendo Senderos Que Duran').first().click()
  await page.getByRole('heading', { name: /Construyendo Senderos Que Duran/ }).waitFor()
  // Mobile keeps the transcript opt-in; open it when the toggle is the visible affordance.
  const toggle = page.getByTestId('transcript-toggle')
  await toggle.waitFor({ state: 'attached', timeout: 15_000 })
  if (await toggle.isVisible()) await toggle.click()
}

test('a translated episode reads in English by default and switches to the original', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'transcript-language', testInfo)
  await openP10(page)

  // D-38: English is the default on every surface, so the ENGLISH render is what loads — even
  // though the episode was recorded in Spanish and the page title stays in Spanish (D-42).
  await expect(page.getByText(ENGLISH_LINE).first()).toBeVisible({ timeout: 15_000 })

  const control = page.getByTestId('transcript-language-control')
  await expect(control).toBeVisible()
  await expect(page.getByTestId('transcript-lang-en')).toHaveAttribute('aria-pressed', 'true')

  // Switch to the original: the Spanish body replaces the English one in place.
  await page.getByTestId('transcript-lang-source').click()
  await expect(page.getByText(SPANISH_LINE).first()).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('transcript-lang-source')).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByText(ENGLISH_LINE)).toHaveCount(0)

  // And back, so the control is a real toggle rather than a one-way door.
  await page.getByTestId('transcript-lang-en').click()
  await expect(page.getByText(ENGLISH_LINE).first()).toBeVisible({ timeout: 15_000 })
})

test('an English-native episode offers no language control', async ({ page }, testInfo) => {
  await signInIsolated(page, 'transcript-language-native', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await page.getByRole('heading', { name: /Index Investing Without the Myths/ }).waitFor()
  const toggle = page.getByTestId('transcript-toggle')
  await toggle.waitFor({ state: 'attached', timeout: 15_000 })
  if (await toggle.isVisible()) await toggle.click()
  await expect(page.getByText(/Index funds are not a strategy/).first()).toBeVisible({
    timeout: 15_000,
  })
  // No second rendering exists, so a toggle here would be a control with one position.
  await expect(page.getByTestId('transcript-language-control')).toHaveCount(0)
})

test('the episode is findable by searching in English AND in its own language', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'transcript-language-search', testInfo)

  for (const query of ['trail building drainage', 'construcción de senderos drenaje']) {
    await page.goto(`/search?q=${encodeURIComponent(query)}`)
    // The Spanish show, reachable from a query in either language: English through the analysis
    // layer, Spanish through the vector-less `segments_nonen` tier that the BM25 leg reads.
    await expect(
      page.getByText(/Sesiones de Sendero|Construyendo Senderos Que Duran/).first(),
    ).toBeVisible({ timeout: 20_000 })
  }
})
