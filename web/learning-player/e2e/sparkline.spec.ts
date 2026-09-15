import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Sparkline — the shared inline chart primitive (trend chips, Profile activity). REAL API over the
 * committed corpus, NO mocks. It is `aria-hidden` and decorative, so there is no behaviour to drive;
 * what matters is that it draws a real path FROM DATA rather than rendering an empty `d` — the
 * failure a numbers-to-path helper actually has. Asserted on the Home discovery rows, where each
 * topic/person carries an inline sparkline inside the `discovery-row`.
 */
test('discovery rows draw a real sparkline path from the corpus, not an empty one', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'sparkline', testInfo)
  await page.goto('/')
  // Topics tab is the default; discovery-row rows carry an inline Sparkline each.
  await expect(page.getByTestId('discovery-tab-topic')).toBeVisible()

  const row = page.getByTestId('discovery-row').first()
  await expect(row).toBeVisible()

  const line = row.getByTestId('sparkline-line')
  await expect(line).toHaveCount(1)
  // A path built from real monthly values has drawing commands; an empty series would be "" or a
  // single move. Require an actual line segment (an "L" command), i.e. at least two points.
  const d = await line.getAttribute('d')
  expect(d, 'sparkline drew no path from the series').toMatch(/L/)
})
