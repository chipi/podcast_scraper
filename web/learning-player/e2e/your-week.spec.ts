import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Your Week — the in-app personal digest section on Home. REAL API over the committed validation
 * corpus (tests/fixtures/app-validation-corpus/v3), NO mocks.
 *
 * Populated content is per-user AND spaced-repetition/heard/follow-gated, so it is NOT deterministic
 * in a fresh e2e run — that path is proven at the tier-3 route test (test_app_your_week_corpus_e2e.py,
 * real assembler + corpus) and the component unit tests (YourWeek.test.ts). What the browser locks
 * here is the contract those layers can't: the section mounts, calls the REAL API, and HIDES cleanly
 * when there's nothing to show — no empty shell, no error — signed-out and for a brand-new user.
 */

test('Your Week is absent when signed out', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText("Find any moment you've heard.")).toBeVisible() // home rendered
  await expect(page.getByTestId('your-week')).toHaveCount(0)
})

test('Your Week stays hidden for a fresh signed-in user with no activity', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'your-week', testInfo) // asserts the signed-in state (Sign out visible)
  await page.goto('/')
  // No captures, heard episodes, or follows yet → nothing due → the section must not render.
  await expect(page.getByTestId('your-week')).toHaveCount(0)
})
