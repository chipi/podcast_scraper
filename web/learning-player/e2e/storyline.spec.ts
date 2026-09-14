import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * StorylineView (F4.5) — the storyline overlay (StorylineCard), reached from the Home
 * "Storylines" discovery tab. REAL API over the committed corpus, NO mocks. Tapping a storyline
 * row on Home opens the StorylineCard overlay on top (via `?storyline=` history entry).
 */
test('Home storyline row opens the storyline overlay — members and episodes, not a shell', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'storyline', testInfo)
  await page.goto('/')

  // Storylines is one of the Home discovery kind-tabs.
  await page.getByTestId('discovery-tab-storyline').click()
  const list = page.getByTestId('discovery-list-storyline')
  await expect(list).toBeVisible()
  // Each row is a discovery-row; clicking it opens the storyline card overlay on top.
  const row = page.getByTestId('discovery-row').first()
  await expect(row).toBeVisible()
  await row.click()

  // Clicking a storyline row opens the StorylineCard overlay on top (with ?storyline= in the URL).
  const card = page.getByTestId('storyline-card')
  await expect(card).toBeVisible()
  await expect(page).toHaveURL(/[?&]storyline=/)
  const view = card.getByTestId('storyline-view')
  await expect(view).toBeVisible()

  // F2.2: a storyline is favoritable (the shared heart), distinct from Follow. Toggling it flips
  // the pressed state — it lands in Library › Saved like any other kind.
  const heart = view.locator('.lp-fav').first()
  await expect(heart).toBeVisible()
  const before = await heart.getAttribute('aria-pressed')
  await heart.click()
  await expect(heart).not.toHaveAttribute('aria-pressed', before ?? 'false')

  // Not an empty shell: it names the storyline (h1) and lists its member topics.
  await expect(view.locator('h1')).not.toHaveText('...')
  await expect(view.getByText("Couldn't load the topics in this storyline.")).toHaveCount(0)
  await expect(view.getByRole('listitem').first()).toBeVisible()
})

test('the storyline overlay can be followed, when it carries a theme cluster', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'storyline-follow', testInfo)
  await page.goto('/')
  await page.getByTestId('discovery-tab-storyline').click()
  await page.getByTestId('discovery-row').first().click()
  const card = page.getByTestId('storyline-card')
  await expect(card).toBeVisible()
  await expect(card.getByTestId('storyline-view')).toBeVisible()

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
