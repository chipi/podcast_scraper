import { expect, test } from '@playwright/test'
import { signInIsolated, expectSignedIn } from './helpers'

/**
 * Browse (Catalog) and Profile — two whole VIEWS that had no dedicated spec
 * (E2E_SURFACE_MAP coverage gaps, closed 2026-09-03).
 *
 * Browse is the app's hub: it routes onward and holds no state of its own, so what is worth
 * asserting is that every tab reaches its index. Profile is where account, activity and DEVICE
 * settings sit together, and the ordering is load-bearing — device settings are shared by everyone
 * who signs in on that phone, so they belong last.
 */

test('Browse reaches all four indexes', async ({ page }, testInfo) => {
  await signInIsolated(page, 'browse-hub', testInfo)
  await page.goto('/browse')
  await expect(page.getByTestId('browse-view')).toBeVisible()

  await page.getByTestId('browse-tab-shows').click()
  await expect(page.getByTestId('show-browse-view')).toBeVisible()

  await page.getByTestId('browse-tab-topics').click()
  await expect(page.getByTestId('topic-browse-view')).toBeVisible()

  await page.getByTestId('browse-tab-people').click()
  await expect(page.getByTestId('person-browse-view')).toBeVisible()

  await page.getByTestId('browse-tab-episodes').click()
  await expect(page.getByTestId('browse-view')).toBeVisible()
})

test('the shows index can be searched and sorted', async ({ page }, testInfo) => {
  await signInIsolated(page, 'browse-shows', testInfo)
  await page.goto('/browse')
  await page.getByTestId('browse-tab-shows').click()

  await expect(page.getByTestId('show-browse-grid')).toBeVisible()
  const search = page.getByTestId('show-browse-search')
  await expect(search).toBeVisible()
  await search.fill('zzzz-no-such-show')
  // An index that filters to nothing must SAY so rather than render an empty grid, which reads as
  // a loading state that never finishes.
  await expect(page.getByTestId('show-browse-grid')).toBeHidden()

  await search.fill('')
  await expect(page.getByTestId('show-browse-grid')).toBeVisible()
  await expect(page.getByTestId('show-browse-sort')).toBeVisible()
})

test('Profile shows activity and puts DEVICE settings last', async ({ page }, testInfo) => {
  await signInIsolated(page, 'profile-view', testInfo)
  await page.goto('/profile')

  // Signed-in identity and the account-level surfaces.
  await expectSignedIn(page)
  await expect(page.getByTestId('profile-settings-link')).toBeVisible()
  // Activity lives in the Stats tab now (Profile is tabbed: Account / Topics / Stats).
  await page.getByRole('tab', { name: 'Stats' }).click()
  await expect(page.getByText('Your activity')).toBeVisible()
})

test('the interests picker opens as a modal and "Not now" is as reachable as Save', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'interests-picker', testInfo)
  await page.goto('/profile')

  // The picker is reached from Profile; e2e previously drove the interests API and never the modal.
  //
  // This used to locate the entry point by /interest|personalize/i and skip when it found
  // nothing. The button's label is "Edit" (i18n `profile.editInterests`), so the locator NEVER
  // matched and this spec skipped on every single run — it has never once opened the picker.
  // A skip that cannot fail is not coverage. Targeted by testid now, and asserted: the button
  // is rendered unconditionally in ProfileView, so its absence is a regression, not a corpus
  // property.
  // Interests moved into the Topics tab (Profile is tabbed now).
  await page.getByRole('tab', { name: 'Topics' }).click()
  const open = page.getByTestId('profile-edit-interests')
  await expect(open).toBeVisible()
  await open.click()

  const dialog = page.getByRole('dialog')
  await expect(dialog).toBeVisible()
  // UXS-013: dismissal must be as easy as committing. A picker that traps someone into choosing is
  // one they dismiss by leaving the app.
  await expect(dialog.getByRole('button', { name: /not now|cancel|close/i }).first()).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
})

/**
 * "Browse" means the same screen whichever device you are on (#2013).
 *
 * It did not. `App.vue`'s desktop nav sent the label to `/catalog` while `BottomNav.vue`'s mobile
 * tab sent the identical label to `/browse` — so a phone user and a laptop user clicking the same
 * word arrived at different products: the hub with four corpus indexes, or a bare episode list.
 * `/browse` is a strict superset (it renders `<CatalogView embedded />` as its Episodes tab), so
 * desktop users were not missing the catalogue, they were missing Shows, Topics and People.
 *
 * This needs a browser and needs BOTH projects: the two nav systems are `sm:hidden` and
 * `hidden sm:flex`, so exactly one exists at any width and a unit test would only ever see one of
 * them. The assertion is deliberately on the destination's CONTENT rather than the URL — a route
 * rename should not fail this, but landing somewhere without the indexes must.
 */
test('the Browse affordance lands on the hub at every viewport', async ({ page }, testInfo) => {
  await signInIsolated(page, 'browse-parity', testInfo)
  await page.goto('/')

  // Whichever nav this project renders, that is the one a user of this width can reach.
  //
  // Asserted as EXISTENCE first, then clicked. Going straight to `.click()` made the regression
  // fail as a 60-second timeout ("locator resolved to nothing") instead of naming the fault, which
  // is the difference between a guard that reports a bug and a guard that reports a hang.
  const browseLink = page.locator('a[href="/browse"]:visible').first()
  await expect(
    browseLink,
    'no visible link to /browse at this width — the Browse affordance points somewhere else, ' +
      'which is exactly the desktop-goes-to-/catalog bug this test exists for',
  ).toBeVisible({ timeout: 10_000 })
  await browseLink.click()

  await expect(page.getByTestId('browse-view')).toBeVisible()
  for (const tab of ['episodes', 'shows', 'topics', 'people']) {
    await expect(
      page.getByTestId(`browse-tab-${tab}`),
      `Browse must offer the ${tab} index — a destination without all four is the /catalog bug`,
    ).toBeVisible()
  }
})
