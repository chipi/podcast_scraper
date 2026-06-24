import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

/**
 * The test that would have caught the transcript_file_path bug — REAL API, REAL fixture
 * corpus, NO mocks. Drives catalog → player and asserts the transcript text actually renders
 * (i.e. the real metadata → /segments → client path works), plus the Knowledge Panel and an
 * axe a11y pass on both surfaces.
 */
test('home → player renders the real transcript + insights (no mocks)', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Learning Player')).toBeVisible()

  // a11y on Home.
  const homeAxe = await new AxeBuilder({ page }).analyze()
  expect(homeAxe.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious')).toEqual([])

  // Open the episode from What's new (real navigation; slug computed by the backend).
  await page.getByText('How Sleep Consolidates Memory').first().click()
  await expect(page.getByRole('heading', { name: 'How Sleep Consolidates Memory' })).toBeVisible()

  // THE regression assertion: the transcript renders from the real corpus (transcript_file_path
  // → /segments). If the metadata key is wrong, this is empty ("Transcript pending").
  await expect(page.getByText('The hippocampus replays sequences during sleep.')).toBeVisible()

  // Knowledge Panel: open and confirm a grounded insight from the real GI artifact.
  await page.getByRole('button', { name: /insights/i }).first().click()
  await expect(page.getByText('Sleep spindles gate memory transfer to the cortex.')).toBeVisible()

  const playerAxe = await new AxeBuilder({ page }).analyze()
  expect(playerAxe.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious')).toEqual([])
})
