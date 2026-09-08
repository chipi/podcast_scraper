import { expect, test } from '@playwright/test'
import { expectSignedIn } from '../helpers'

/**
 * Tier-3 — recall (search with scope=mine) against a real backend +
 * real corpus.
 *
 * The consumer app's "My corpus" toggle re-runs search against the
 * heard∪captured set (PRD-041 §Recall). This walk drives:
 *   1. Perform a search with scope=all — confirm the everything path works
 *   2. Toggle to scope=mine — confirm the honest-zero-coverage state
 *   3. Screenshots both.
 */

test('operator recall: scope=all vs scope=mine on real search', async ({ page }) => {
  await page.goto('/api/app/auth/login?as=tier3-app-recall')
  await expectSignedIn(page)

  await page.goto('/search')
  await page.waitForLoadState('networkidle')
  await page.screenshot({ path: 'validation-results/recall-01-empty.png', fullPage: true })

  // Type a generic query the real corpus is very likely to have something for.
  const searchInput = page.getByRole('searchbox').first()
  await expect(searchInput).toBeVisible()
  await searchInput.fill('the')
  await searchInput.press('Enter')
  await page.waitForLoadState('networkidle')
  await page.screenshot({
    path: 'validation-results/recall-02-all-results.png',
    fullPage: true,
  })

  // Toggle to "My corpus" — signed-out heard∪captured is empty, so this
  // exercises the honest-zero-coverage UX (PRD-041) rather than fake
  // results. The regression signal: the toggle exists, is reachable, and
  // updates the results surface.
  // The locator has been wrong twice now, and both times the spec went quiet rather than red.
  // First it looked for a `radio`/`button` named /my corpus/i and matched nothing, so the scope
  // half of a spec CALLED "scope=all vs scope=mine" simply never ran. It was corrected to the
  // `role="tab"` pair that SearchView rendered — and #1594 item 7 then made that pair a RADIOGROUP
  // (`role="radio"`, `aria-checked`), because the scope re-queries one results region rather than
  // switching between panels. So it is a radio again, for a different reason.
  //
  // Result CONTENT is deliberately not asserted: this tier installs no ML extras, so the index
  // cannot be queried and the surface answers "Search is temporarily unavailable" — asserting
  // hits here would fail for a missing optional dependency rather than for a regression. The
  // scope CONTROL is independent of the index, and that is what is asserted.
  const everything = page.getByRole('radio', { name: 'Everything' })
  const mine = page.getByRole('radio', { name: 'My listening' })
  await expect(everything).toBeVisible()
  await expect(mine).toBeVisible()
  await expect(everything).toHaveAttribute('aria-checked', 'true')

  await mine.click()
  await page.waitForLoadState('networkidle')
  // Selection moved AND the scope is reflected in the URL, so a shared/reloaded link keeps it.
  await expect(mine).toHaveAttribute('aria-checked', 'true')
  await expect(everything).toHaveAttribute('aria-checked', 'false')
  await expect(page).toHaveURL(/scope=mine/)
  await page.screenshot({
    path: 'validation-results/recall-03-mine-results.png',
    fullPage: true,
  })
})
