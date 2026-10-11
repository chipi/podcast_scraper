import { expect, test, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * A link inside an entity card LEAVES: it lands on its destination, not back where the card was
 * opened. The trap (2026-10-11, found by the iOS native tier's HostShowLinkTests): a link that also
 * closed its sheet took the ✕ path, whose `router.back()` ran before the route settled and undid the
 * navigation — "Host of <show>" and "Search the library" left the listener on the episode. EpisodeRow
 * hit the same trap on 2026-09-16. Navigating is what closes the sheet (its query disappears).
 */
const EPISODE = '/episode/p09-a4bbb5dde3'

async function openFromBrief(page: Page, name: RegExp): Promise<void> {
  await page.goto(EPISODE)
  await page.getByTestId('player-open-insights').click()
  // The chips render after the panel opens; past the first few they fold behind "+N more".
  await expect(page.getByTestId('kp-topic-chip').first()).toBeVisible()
  const chip = page.getByRole('button', { name }).first()
  const more = page.getByTestId('kp-tags-more')
  if (!(await chip.isVisible()) && (await more.isVisible())) await more.click()
  await chip.click()
}

test('"Host of" on a person card opens the show', async ({ page }, testInfo) => {
  await signInIsolated(page, 'card-links-host', testInfo)
  await openFromBrief(page, /Open Sam/)
  const link = page.getByTestId('ec-host-show-link').last()
  const show = await link.innerText()
  await link.click()
  await expect(page).toHaveURL(/\/podcast\/[^/?]+$/)
  await expect(page.getByRole('heading', { name: show }).first()).toBeVisible()
})

test("a person card's Search the library runs the search", async ({ page }, testInfo) => {
  await signInIsolated(page, 'card-links-person-search', testInfo)
  await openFromBrief(page, /Open Sam/)
  await page.getByTestId('ec-search-library').last().click()
  await expect(page).toHaveURL(/\/search\?q=Sam/)
})

test("a topic card's Search the library runs the search", async ({ page }, testInfo) => {
  await signInIsolated(page, 'card-links-topic-search', testInfo)
  await openFromBrief(page, /Open risk management/)
  await page.getByTestId('ec-search-library').last().click()
  await expect(page).toHaveURL(/\/search\?q=risk/)
})
