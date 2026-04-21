import { expect, test } from '@playwright/test'
import { loadGraphViaFilePicker } from './helpers'

test.describe('Graph gesture overlay', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.removeItem('ps_graph_hints_seen')
    })
  })

  test('shows on first graph load and Got it sets storage', async ({ page }) => {
    await loadGraphViaFilePicker(page)
    const overlay = page.getByTestId('graph-gesture-overlay')
    await expect(overlay).toBeVisible()
    await page.getByTestId('graph-gesture-overlay-dismiss').click()
    await expect(overlay).toBeHidden()
    expect(await page.evaluate(() => localStorage.getItem('ps_graph_hints_seen'))).toBe('1')
  })

  test('backdrop click outside card dismisses', async ({ page }) => {
    await loadGraphViaFilePicker(page)
    const overlay = page.getByTestId('graph-gesture-overlay')
    await expect(overlay).toBeVisible()
    await overlay.click({ position: { x: 4, y: 4 } })
    await expect(overlay).toBeHidden()
    expect(await page.evaluate(() => localStorage.getItem('ps_graph_hints_seen'))).toBe('1')
  })

  test('Gestures reopen shows overlay without clearing storage flag', async ({ page }) => {
    await loadGraphViaFilePicker(page)
    await page.getByTestId('graph-gesture-overlay-dismiss').click()
    expect(await page.evaluate(() => localStorage.getItem('ps_graph_hints_seen'))).toBe('1')
    await page.getByTestId('graph-gesture-overlay-reopen').click()
    await expect(page.getByTestId('graph-gesture-overlay')).toBeVisible()
    await page.getByTestId('graph-gesture-overlay-dismiss').click()
    expect(await page.evaluate(() => localStorage.getItem('ps_graph_hints_seen'))).toBe('1')
  })
})

test.describe('Graph gesture overlay persistence across reload', () => {
  test('does not auto-open after dismiss, reload, and load graph again', async ({ page }) => {
    await page.addInitScript(() => {
      if (sessionStorage.getItem('ps_e2e_gesture_persist') === '1') {
        return
      }
      localStorage.removeItem('ps_graph_hints_seen')
    })

    await loadGraphViaFilePicker(page)
    await page.getByTestId('graph-gesture-overlay-dismiss').click()
    expect(await page.evaluate(() => localStorage.getItem('ps_graph_hints_seen'))).toBe('1')

    await page.evaluate(() => sessionStorage.setItem('ps_e2e_gesture_persist', '1'))
    await page.reload()
    await loadGraphViaFilePicker(page)
    await expect(page.getByTestId('graph-gesture-overlay')).toBeHidden()

    await page.evaluate(() => {
      sessionStorage.removeItem('ps_e2e_gesture_persist')
      localStorage.removeItem('ps_graph_hints_seen')
    })
  })
})
