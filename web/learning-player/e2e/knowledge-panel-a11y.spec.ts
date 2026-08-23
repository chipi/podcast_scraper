import { expect, test } from '@playwright/test'

/**
 * The Knowledge Panel sheet is a real dialog (S9).
 *
 * It is the entry point to the learning features — the reason this product is not a podcast player.
 * On mobile it covers the whole screen, and it shipped as a plain `<div>` that never said so: no
 * `role="dialog"`, no focus trap, no Escape, and the opener was removed from the DOM on open so
 * focus fell to `<body>`. A keyboard or screen-reader user could open the differentiator and then
 * be unable to move through it or leave it.
 *
 * These assertions are e2e rather than unit because the behaviour belongs to the browser: happy-dom
 * does not implement the top layer, `::backdrop`, inertness, or `showModal()`'s focus semantics, so
 * a unit test here would assert my mock rather than the platform.
 */

/** Open the panel from the ✦ control, which is the labelled entry point #1595 built. */
async function openPanel(page: import('@playwright/test').Page) {
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  const opener = page.getByTestId('player-open-insights')
  await expect(opener).toBeVisible()
  await opener.click()
  return opener
}

test.describe('Knowledge Panel dialog semantics', () => {
  test('opens as a modal on mobile, and Escape closes it', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'mobile-chrome', 'modal mode is the mobile presentation')

    await openPanel(page)
    const panel = page.getByTestId('knowledge-panel')
    await expect(panel).toBeVisible()

    // `open` alone is a non-modal dialog. Modality is what supplies the focus trap and inert
    // background, and it is exactly what the old <div> lacked.
    expect(await panel.evaluate((d) => (d as HTMLDialogElement).matches(':modal'))).toBe(true)

    await page.keyboard.press('Escape')
    await expect(panel).toBeHidden()
  })

  test('traps focus inside the sheet, so Tab cannot reach the page behind it', async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== 'mobile-chrome', 'modal mode is the mobile presentation')

    await openPanel(page)
    const panel = page.getByTestId('knowledge-panel')
    await expect(panel).toBeVisible()

    // Tab a good number of times. Focus may pass through <body> — that is Chrome's neutral wrap
    // point at the end of a modal's focus cycle, and it is not a control anyone can operate. What
    // must never happen is focus landing on an interactive element of the covered page: with the
    // old plain <div>, the header links took focus within a few presses and the user was operating
    // controls they could not see.
    for (let i = 0; i < 25; i++) {
      await page.keyboard.press('Tab')
      const where = await page.evaluate(() => {
        const d = document.querySelector('[data-testid="knowledge-panel"]')
        const a = document.activeElement as HTMLElement | null
        if (!a || a === document.body) return 'neutral'
        return d?.contains(a) ? 'inside' : `OUTSIDE:${a.tagName}.${a.className.slice(0, 40)}`
      })
      expect(where, `after ${i + 1} Tab press(es)`).not.toContain('OUTSIDE')
    }
  })

  test('the page behind the sheet is inert — its controls cannot be focused at all', async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== 'mobile-chrome', 'modal mode is the mobile presentation')

    await openPanel(page)
    await expect(page.getByTestId('knowledge-panel')).toBeVisible()

    // Inertness is the property that makes the sheet a dialog rather than a big div: even a direct
    // programmatic focus() on a background control must do nothing while the modal is up.
    const tookFocus = await page.evaluate(() => {
      const behind = document.querySelector('header a') as HTMLElement | null
      if (!behind) return 'no-background-control-found'
      behind.focus()
      return document.activeElement === behind ? 'focused' : 'inert'
    })
    expect(tookFocus).toBe('inert')
  })

  test('returns focus to the control that opened it', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'mobile-chrome', 'modal mode is the mobile presentation')

    const opener = await openPanel(page)
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('knowledge-panel')).toBeHidden()

    // The opener is `v-if="!panelOpen"`, so it does not exist when the browser would restore focus
    // itself — without the explicit restore, focus lands on <body> and the user loses their place.
    await expect(opener).toBeFocused()
  })

  test('is a docked rail on desktop, NOT a modal — nothing is covered there', async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name !== 'desktop-chrome', 'rail mode is the desktop presentation')

    await openPanel(page)
    const panel = page.getByTestId('knowledge-panel')
    await expect(panel).toBeVisible()

    // Trapping focus beside the player would be wrong: the transcript is still visible and must
    // stay reachable by keyboard.
    expect(await panel.evaluate((d) => (d as HTMLDialogElement).matches(':modal'))).toBe(false)
    expect(await panel.evaluate((d) => (d as HTMLDialogElement).open)).toBe(true)
  })
})
