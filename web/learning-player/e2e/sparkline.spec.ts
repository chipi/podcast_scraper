import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Sparkline — the shared inline chart primitive (trend chips, Profile activity). REAL API over the
 * committed corpus, NO mocks. It is `aria-hidden` and decorative, so there is no behaviour to drive;
 * what matters is that it draws a real path FROM DATA rather than rendering an empty `d` — the
 * failure a numbers-to-path helper actually has. Asserted on the Home "Sparklines" trend rows, where
 * each rising topic carries one.
 */
test('trend rows draw a real sparkline path from the corpus, not an empty one', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'sparkline', testInfo)
  await page.goto('/')
  await page.getByTestId('discovery-tab-trending').click()

  const row = page.getByTestId('trend-spark-row').first()
  await expect(row).toBeVisible()

  const line = row.getByTestId('sparkline-line')
  await expect(line).toHaveCount(1)
  // A path built from real monthly values has drawing commands; an empty series would be "" or a
  // single move. Require an actual line segment (an "L" command), i.e. at least two points.
  const d = await line.getAttribute('d')
  expect(d, 'sparkline drew no path from the series').toMatch(/L/)
})
