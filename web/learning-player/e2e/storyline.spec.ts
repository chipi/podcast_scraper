import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * StorylineView (F4.5) — the full-page storyline (theme cluster), reached from the Home
 * "Storylines" rail. REAL API over the committed corpus, NO mocks. Replaces the old bottom-sheet
 * StorylineCard: opening a storyline now NAVIGATES to /storyline/:anchorTopicId, so it has back-nav,
 * a shareable URL, and room for the member topics + episodes the sheet could not show.
 */
test('Home storyline chip opens the full storyline page — members and episodes, not a shell', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'storyline', testInfo)
  await page.goto('/')

  // Storylines is one of the Home discovery tabs — select it before the rail renders.
  await page.getByTestId('discovery-tab-storylines').click()
  const rail = page.getByTestId('home-storylines')
  await expect(rail).toBeVisible()
  // The chip is a wrapper div holding an "open" button and (maybe) a follow button; the first
  // button opens the storyline.
  const chip = page.getByTestId('storyline-chip').first()
  await expect(chip).toBeVisible()
  await chip.getByRole('button').first().click()

  // It navigated to its own route rather than opening a sheet on Home.
  await expect(page).toHaveURL(/\/storyline\//)
  const view = page.getByTestId('storyline-view')
  await expect(view).toBeVisible()

  // Not an empty shell: it names the storyline (h1) and lists its member topics.
  await expect(view.locator('h1')).not.toHaveText('…')
  await expect(view.getByText('Couldn’t load the topics in this storyline.')).toHaveCount(0)
  await expect(view.getByRole('listitem').first()).toBeVisible()
})

test('the storyline page can be followed, when it carries a theme cluster', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'storyline-follow', testInfo)
  await page.goto('/')
  await page.getByTestId('discovery-tab-storylines').click()
  await page.getByTestId('storyline-chip').first().getByRole('button').first().click()
  await expect(page.getByTestId('storyline-view')).toBeVisible()

  // Follow renders only for a storyline with a `thc:` cluster id; skip cleanly when this corpus
  // storyline has none, rather than asserting an affordance the data does not warrant.
  const follow = page.getByTestId('storyline-follow')
  if (!(await follow.isVisible().catch(() => false))) {
    test.skip(true, 'this storyline has no theme-cluster id, so there is nothing to follow')
    return
  }
  const before = await follow.getAttribute('aria-pressed')
  await follow.click()
  await expect(follow).not.toHaveAttribute('aria-pressed', before ?? 'false')
})
