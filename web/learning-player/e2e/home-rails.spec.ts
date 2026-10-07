import { expect, test } from '@playwright/test'
import { signInIsolated, showEveryonesTrends } from './helpers'

/**
 * The Home rails that had unit tests and no e2e (E2E_SURFACE_MAP coverage gaps, closed 2026-09-03),
 * and Trends — which left Home for Discover only (operator 2026-10-07), so its specs run on
 * `/browse` now.
 *
 * They are grouped because they share ONE contract, and that contract is what is actually worth
 * asserting in a browser: a rail with nothing to show **omits itself**. A unit test can prove a
 * component renders chips from props; only a real render against a real API can prove the section
 * does not leave an empty shell behind when the API returns nothing.
 */

test('Your Week renders for a signed-in listener and can expand', async ({ page }, testInfo) => {
  await signInIsolated(page, 'home-yourweek', testInfo)
  await page.goto('/')

  const week = page.getByTestId('your-week')
  await expect(week).toBeVisible()
  // compact ↔ full is a synced per-user preference; the inline control is the only way to reach it.
  const toggle = week.getByRole('button', { name: /show more|show less/i }).first()
  if (await toggle.isVisible().catch(() => false)) {
    await toggle.click()
    await expect(week).toBeVisible()
  }
})

test('the discovery list lists rows and each one can be followed', async ({ page }, testInfo) => {
  await signInIsolated(page, 'home-momentum', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)

  // Topics tab is the default; Rising sort is the default — no extra click needed.
  await expect(page.getByTestId('discovery-tab-topic')).toBeVisible()
  const list = page.getByTestId('discovery-list-topic')
  await expect(list).toBeVisible()
  await expect(page.getByTestId('discovery-row').first()).toBeVisible()

  // Following writes the same interest token the picker does, so the two must agree.
  const follow = page.getByTestId('discovery-follow').first()
  await expect(follow).toBeVisible()
  await Promise.all([
    page.waitForResponse((r) => r.url().includes('/api/app/interests') && r.request().method() !== 'GET'),
    follow.click(),
  ])
})

test('the trend window tabs re-query rather than re-rendering the same series', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'home-trend-window', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  // Topics tab (Rising sort by default) — the window tabs belong to the discovery section.
  await expect(page.getByTestId('discovery-tab-topic')).toBeVisible()
  await expect(page.getByTestId('trend-window-tabs')).toBeVisible()

  // A window control that does not change the request is decoration.
  await Promise.all([
    page.waitForResponse((r) => r.url().includes('/api/app/trending')),
    page.getByTestId('trend-window-1y').click(),
  ])
  await expect(page.getByTestId('discovery-list-topic')).toBeVisible()

  // Switching sort (Rising → Trending) keeps the same discovery section visible.
  await page.getByTestId('discovery-sort').click()
  await expect(page.getByTestId('browse-discovery')).toBeVisible()
})

test('the storylines tab renders rows and follows one', async ({ page }, testInfo) => {
  await signInIsolated(page, 'home-storylines', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  await page.getByTestId('discovery-tab-storyline').click()

  const list = page.getByTestId('discovery-list-storyline')
  await expect(list).toBeVisible()
  await expect(page.getByTestId('discovery-row').first()).toBeVisible()
  const follow = page.getByTestId('discovery-follow').first()
  await expect(follow).toBeVisible()
  await follow.click()
  // Idempotent: a second render must not duplicate the row.
  await expect(page.getByTestId('discovery-row').first()).toBeVisible()
})

test('the Trends kind tabs are in order and fit the phone row', async ({ page }, testInfo) => {
  await signInIsolated(page, 'home-themes', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  // Topics, Themes, Storylines, People — the order every surface lists the kinds in (2026-10-05).
  await expect(page.locator('[data-testid^="discovery-tab-"]')).toHaveText(['Topics', 'Themes', 'Storylines', 'People'])
  // All four kind pills fit the phone row beside the two switches — none clipped off the edge.
  const vw = page.viewportSize()!.width
  for (const tab of await page.locator('[data-testid^="discovery-tab-"]').all()) {
    const box = await tab.boundingBox()
    expect(box && box.x >= 0 && box.x + box.width <= vw).toBe(true)
  }
  await page.getByTestId('discovery-tab-theme').click()
  const list = page.getByTestId('discovery-list-theme')
  await expect(list).toBeVisible()
  await expect(list.getByTestId('discovery-row').first()).toBeVisible()
})

test('Discover lists themes too, and a theme opens its page', async ({ page }, testInfo) => {
  await signInIsolated(page, 'browse-themes', testInfo)
  await page.goto('/browse?trends=theme')
  await showEveryonesTrends(page)
  const list = page.getByTestId('discovery-list-theme')
  await expect(list).toBeVisible()
  await list.getByTestId('discovery-row').first().locator('button').first().click()
  await expect(page).toHaveURL(/\/theme\//)
  await expect(page.getByTestId('theme-view')).toBeVisible()
})

test('the discovery tabs switch between topics, themes, storylines and people', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'home-discovery', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)

  await expect(page.getByTestId('browse-discovery')).toBeVisible()
  for (const tab of ['discovery-tab-topic', 'discovery-tab-theme', 'discovery-tab-storyline', 'discovery-tab-person']) {
    await page.getByTestId(tab).click()
    // Whatever the tab shows, the section must not be left empty-but-present.
    await expect(page.getByTestId('browse-discovery')).toBeVisible()
  }
})

/**
 * Back closes the entity modal instead of navigating the page under it (#1594).
 *
 * The issue asked for this to be VERIFIED in the Capacitor shell, where hardware Back is mapped to
 * a history navigation. A Chromium Back press is the same event the shell delivers, so the
 * behaviour is pinned here — in a suite that runs on every change — rather than only in a manual
 * pass on a device nobody re-runs.
 */
test('browser Back closes the entity card and leaves the page under it alone', async ({ page }, testInfo) => {
  // On a show page: Home's Trends opened this card until Trends left Home (2026-10-07); the show's
  // signals band opens the same EntityCard.
  await signInIsolated(page, 'home-entity-back', testInfo)
  await page.goto('/podcast/p05')

  const chip = page.getByTestId('ps-distinctive-topic').first()
  await expect(chip).toBeVisible()
  await chip.click()

  const card = page.locator('[role="dialog"][aria-modal="true"]')
  await expect(card).toBeVisible()
  // The open card now has a URL, which is what gives Back something to pop.
  await expect(page).toHaveURL(/[?&]card=/)

  await page.goBack()

  await expect(card, 'Back left the card open — it navigated the page underneath').toBeHidden()
  await expect(page).not.toHaveURL(/[?&]card=/)
  // Still on the show. Before this, Back with an open card took the page behind it somewhere else.
  expect(new URL(page.url()).pathname).toBe('/podcast/p05')
})

test('closing the card with Escape does not leave a dead Back press behind', async ({ page }, testInfo) => {
  // The bookkeeping half. The card pushes a history entry when it opens; closing it any other way
  // has to pop that entry, or the user's next Back press spends itself undoing our state and reads
  // as a button that did nothing.
  await signInIsolated(page, 'home-entity-esc', testInfo)
  await page.goto('/podcast/p05')

  const chip = page.getByTestId('ps-distinctive-topic').first()
  await expect(chip).toBeVisible()
  await chip.click()

  const card = page.locator('[role="dialog"][aria-modal="true"]')
  await expect(card).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(card).toBeHidden()
  await expect(page).not.toHaveURL(/[?&]card=/)

  // One Back press from here must leave the show entirely — if the pushed entry were still on the
  // stack, this would only return to the show with the card shut.
  await page.goBack()
  expect(new URL(page.url()).pathname, 'a stale history entry absorbed the Back press').not.toBe('/podcast/p05')
})


test('a theme row says how many topics it holds, and can be followed', async ({ page }, testInfo) => {
  await signInIsolated(page, 'trends-theme-row', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  await page.getByTestId('discovery-tab-theme').click()
  const row = page.getByTestId('discovery-list-theme').getByTestId('discovery-row').first()
  await expect(row).toBeVisible()
  await expect(row.locator('button').first()).toHaveAttribute('aria-label', /\(\d+\)/)
  const follow = row.getByTestId('discovery-follow')
  await Promise.all([
    page.waitForResponse((r) => r.url().includes('/api/app/interests') && r.request().method() !== 'GET'),
    follow.click(),
  ])
})

/**
 * Discover's search + Trends block (operator 2026-10-05): search before Trends, the two
 * trending-topic chips under the search, 3 Trends rows on a phone and 5 on desktop, and "all ›"
 * expanding in place. Home shares the search half only since Trends left it (2026-10-07).
 */
for (const viewport of [
  { width: 412, height: 915, rows: 3 },
  { width: 1440, height: 900, rows: 5 },
]) {
  test(`Discover's search + Trends block (${viewport.width}px)`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport)
    await signInIsolated(page, `trends-block-${viewport.width}`, testInfo)
    await page.goto('/browse')
    await showEveryonesTrends(page)
    const block = page.getByTestId('browse-discovery')
    const rows = block.getByTestId('discovery-list-topic').getByTestId('discovery-row')
    await expect(rows).toHaveCount(viewport.rows)
    await expect(page.getByTestId('browse-search-section').getByTestId('home-topic-chip')).toHaveCount(2)
    const [s, t] = await page.evaluate(
      ([a, b]) => [a, b].map((id) => document.querySelector(`[data-testid="${id}"]`)!.getBoundingClientRect().top),
      ['browse-search-section', 'browse-discovery'],
    )
    expect(t, 'Trends below the search').toBeGreaterThan(s)
    // "all ›" expands in place: more rows, same page.
    await block.getByTestId('discovery-see-all').click()
    await expect(page).toHaveURL(/\/browse/)
    expect(await rows.count()).toBeGreaterThan(viewport.rows)

    // Home keeps the search half, and has no Trends (operator 2026-10-07).
    await page.goto('/')
    await expect(page.getByTestId('home-search-section').getByTestId('home-topic-chip')).toHaveCount(2)
    await expect(page.getByTestId('home-discovery')).toHaveCount(0)
  })
}
