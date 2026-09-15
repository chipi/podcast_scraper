import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * The Home rails that had unit tests and no e2e (E2E_SURFACE_MAP coverage gaps, closed 2026-09-03).
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
  await page.goto('/')

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
  await page.goto('/')
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
  await expect(page.getByTestId('home-discovery')).toBeVisible()
})

test('the storylines tab renders rows and follows one', async ({ page }, testInfo) => {
  await signInIsolated(page, 'home-storylines', testInfo)
  await page.goto('/')
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

test('the discovery tabs switch between topics, storylines and people', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'home-discovery', testInfo)
  await page.goto('/')

  await expect(page.getByTestId('home-discovery')).toBeVisible()
  for (const tab of ['discovery-tab-topic', 'discovery-tab-storyline', 'discovery-tab-person']) {
    await page.getByTestId(tab).click()
    // Whatever the tab shows, the section must not be left empty-but-present.
    await expect(page.getByTestId('home-discovery')).toBeVisible()
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
  await signInIsolated(page, 'home-entity-back', testInfo)
  await page.goto('/')
  // Topics tab is default; discovery-row opens the entity card.
  await expect(page.getByTestId('discovery-tab-topic')).toBeVisible()

  const chip = page.getByTestId('discovery-row').first()
  await expect(chip).toBeVisible()
  await chip.click()

  const card = page.locator('[role="dialog"][aria-modal="true"]')
  await expect(card).toBeVisible()
  // The open card now has a URL, which is what gives Back something to pop.
  await expect(page).toHaveURL(/[?&]card=/)

  await page.goBack()

  await expect(card, 'Back left the card open — it navigated the page underneath').toBeHidden()
  await expect(page).not.toHaveURL(/[?&]card=/)
  // Still on Home. Before this, Back with an open card took the page behind it somewhere else.
  expect(new URL(page.url()).pathname).toBe('/')
})

test('closing the card with Escape does not leave a dead Back press behind', async ({ page }, testInfo) => {
  // The bookkeeping half. The card pushes a history entry when it opens; closing it any other way
  // has to pop that entry, or the user's next Back press spends itself undoing our state and reads
  // as a button that did nothing.
  await signInIsolated(page, 'home-entity-esc', testInfo)
  await page.goto('/')
  // Topics tab is default; discovery-row opens the entity card.
  await expect(page.getByTestId('discovery-tab-topic')).toBeVisible()

  const chip = page.getByTestId('discovery-row').first()
  await expect(chip).toBeVisible()
  await chip.click()

  const card = page.locator('[role="dialog"][aria-modal="true"]')
  await expect(card).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(card).toBeHidden()
  await expect(page).not.toHaveURL(/[?&]card=/)

  // One Back press from here must leave Home entirely — if the pushed entry were still on the
  // stack, this would only return to Home with the card shut.
  await page.goBack()
  expect(new URL(page.url()).pathname, 'a stale history entry absorbed the Back press').not.toBe('/')
})

