import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

/**
 * The test that would have caught the transcript_file_path bug — REAL API over the COMMITTED
 * validation corpus, NO mocks. Drives home → player and asserts the transcript text actually
 * renders (i.e. the real metadata → /segments → client path works), plus the Knowledge Panel
 * insights and an axe a11y pass on both surfaces.
 *
 * Content is deterministically synthesized from the text fixtures (no pipeline): the newest
 * episode is "Index Investing Without the Myths" ("Long Horizon Notes" / fixture p05). Its
 * transcript carries a distinctive line about index funds, and the GI artifact carries grounded
 * insights drawn from the diarized transcript.
 */
test('home → player renders the transcript + insights (no mocks)', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Learning Player')).toBeVisible()

  // a11y on Home.
  const homeAxe = await new AxeBuilder({ page }).analyze()
  expect(homeAxe.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious')).toEqual([])

  // Open the newest episode from What's new (real navigation; slug computed by the backend).
  await page.goto('/podcast/p05') // #1148: reach the episode via its show page (date-independent)
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(
    page.getByRole('heading', { name: /Index Investing Without the Myths/ }),
  ).toBeVisible()

  // THE regression assertion: the transcript renders from the real corpus (transcript_file_path
  // → /segments). If the metadata key is wrong, this is empty ("Transcript pending"). This line
  // is from the committed transcript and is unique to this episode.
  await expect(
    page.getByText(/Index funds are not a strategy/).first(),
  ).toBeVisible()

  // Knowledge Panel: open it (the insights pull-out) and confirm a grounded insight from the GI
  // artifact (the first insight is drawn from the episode's opening turn about index investing).
  await page.getByRole('button', { name: 'Insights' }).first().click()
  await expect(page.getByText(/talking about index investing/).first()).toBeVisible()

  const playerAxe = await new AxeBuilder({ page }).analyze()
  expect(playerAxe.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious')).toEqual([])
})
