import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * OverflowMenu (UXS-014 "Item actions") — the ONE `⋯` menu. REAL API over the committed corpus, NO
 * mocks. Its first host is the player's secondary actions (mark-as-played), so this drives it there:
 * open, act, and confirm the two dismissal paths a fixed teleported panel must honour (Escape and an
 * outside click), which no unit test can prove against real layout.
 */
async function openNewestEpisode(page: import('@playwright/test').Page): Promise<void> {
  await page.goto('/podcast/p05') // reach an episode via its show page (date-independent)
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page.getByTestId('player-controls-sticky')).toBeVisible()
}

test('the player ⋯ menu opens, marks played, and reflects the new state', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'overflow-mark', testInfo)
  await openNewestEpisode(page)

  const trigger = page.getByTestId('overflow-trigger')
  await expect(trigger).toBeVisible()
  await expect(trigger).toHaveAttribute('aria-expanded', 'false')
  await trigger.click()

  const menu = page.getByTestId('overflow-menu')
  await expect(menu).toBeVisible()
  await expect(trigger).toHaveAttribute('aria-expanded', 'true')

  const markPlayed = page.getByTestId('mark-played')
  const wasPlayed = (await markPlayed.textContent())?.includes('unplayed') ?? false
  await markPlayed.click()
  // The menu closes after the item acts, and the toggle now reads the opposite state.
  await expect(menu).toBeHidden()
  await trigger.click()
  const nowUnplayed = (await page.getByTestId('mark-played').textContent())?.includes('unplayed') ?? false
  expect(nowUnplayed).toBe(!wasPlayed)
})

test('the ⋯ menu dismisses on Escape and on an outside click, restoring the page', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'overflow-dismiss', testInfo)
  await openNewestEpisode(page)
  const trigger = page.getByTestId('overflow-trigger')

  // Escape closes and returns focus to the trigger.
  await trigger.click()
  await expect(page.getByTestId('overflow-menu')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('overflow-menu')).toBeHidden()
  await expect(trigger).toBeFocused()

  // An outside pointer closes it too.
  await trigger.click()
  await expect(page.getByTestId('overflow-menu')).toBeVisible()
  await page.mouse.click(5, 5)
  await expect(page.getByTestId('overflow-menu')).toBeHidden()
})
