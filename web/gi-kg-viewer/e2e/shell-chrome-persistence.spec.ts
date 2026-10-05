import { expect, test, type Page, type Request } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from './fixtures'
import {
  liveCorpusRoot,
  loadGraphViaFilePicker,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * UXS-001 / UXS-004 shell chrome: the collapsible left and right rails and the graph bottom bar
 * keep their state across a reload; Alt+B toggles the bottom bar; Escape closes the gesture
 * overlay; Shift+Enter in the search box is a newline, not a submit.
 */

async function landOnDigest(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
  await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
  await expect(page.getByTestId('digest-root')).toBeVisible()
}

async function reloadShell(page: Page): Promise<void> {
  await page.reload()
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
}

test.describe('Rail collapse persistence (live)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('left rail: collapse hides Saved/Recent, survives a reload, the strip reopens it', async ({
    page,
  }) => {
    await landOnDigest(page)
    const toggle = page.getByTestId('left-panel-collapse-toggle')
    await expect(toggle).toHaveAttribute('aria-expanded', 'true')
    await expect(page.getByTestId('left-panel-saved-queries')).toBeVisible()

    await toggle.click()
    await expect(toggle).toHaveAttribute('aria-expanded', 'false')
    await expect(page.getByTestId('left-panel-saved-queries')).toBeHidden()
    await expect(page.getByTestId('left-panel-collapsed-strip')).toBeVisible()

    await reloadShell(page)
    await expect(page.getByTestId('left-panel-collapse-toggle')).toHaveAttribute(
      'aria-expanded',
      'false',
    )
    await expect(page.getByTestId('left-panel-collapsed-strip')).toBeVisible()

    await page.getByTestId('left-panel-collapsed-strip').getByRole('button', { name: 'Saved queries' }).click()
    await expect(page.getByTestId('left-panel-saved-queries')).toBeVisible()
    await expect(page.getByTestId('left-panel-collapse-toggle')).toHaveAttribute(
      'aria-expanded',
      'true',
    )
  })

  test('right rail: collapse leaves the Details strip, survives a reload, Details reopens it', async ({
    page,
  }) => {
    await landOnDigest(page)
    const toggle = page.getByTestId('right-rail-edge-toggle')
    await expect(toggle).toHaveAttribute('aria-expanded', 'true')
    await expect(page.getByTestId('rail-collapsed-subject')).toHaveCount(0)

    await toggle.click()
    await expect(toggle).toHaveAttribute('aria-expanded', 'false')
    await expect(page.getByTestId('rail-collapsed-subject')).toBeVisible()

    await reloadShell(page)
    await expect(page.getByTestId('right-rail-edge-toggle')).toHaveAttribute(
      'aria-expanded',
      'false',
    )
    await page.getByTestId('rail-collapsed-subject').click()
    await expect(page.getByTestId('right-rail-edge-toggle')).toHaveAttribute(
      'aria-expanded',
      'true',
    )
    await expect(page.getByTestId('rail-collapsed-subject')).toHaveCount(0)
  })

  test('Shift+Enter in the search box adds a line; Enter alone searches', async ({ page }) => {
    await landOnDigest(page)
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    const q = page.locator('#search-q')
    let searches = 0
    page.on('request', (r: Request) => {
      if (new URL(r.url()).pathname === '/api/search') searches += 1
    })
    await q.fill('first line')
    await q.press('Shift+Enter')
    await q.pressSequentially('second line')
    await expect(q).toHaveValue('first line\nsecond line')
    expect(searches).toBe(0)

    const req = page.waitForRequest((r) => new URL(r.url()).pathname === '/api/search')
    await q.press('Enter')
    await req
  })
})

test.describe('Graph chrome (offline fixture)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  /**
   * UXS-004 l.150: Escape dismisses only when focus is in the overlay (or nowhere) — with focus in
   * graph chrome such as the Since input it must leave the overlay alone.
   *
   * Fails on 2026-10-05 — app defect, measured: Escape with focus on **Got it** (where the overlay
   * auto-focuses) reaches `window` un-prevented and the overlay stays. `GraphGestureOverlay.vue`
   * declares `overlayRootRef` but no element carries `ref="overlayRootRef"`, so the Escape
   * listener's `if (!root) return` always returns.
   */
  test('Escape closes the gesture overlay from inside it, not from the Since field', async ({
    page,
  }) => {
    await page.addInitScript(() => localStorage.removeItem('ps_graph_hints_seen'))
    await loadGraphViaFilePicker(page)
    const overlay = page.getByTestId('graph-gesture-overlay')
    await expect(overlay).toBeVisible()

    await page.getByTestId('graph-status-since-input').focus()
    await page.keyboard.press('Escape')
    await expect(overlay).toBeVisible()

    await page.getByTestId('graph-gesture-overlay-dismiss').focus()
    await page.keyboard.press('Escape')
    await expect(overlay).toBeHidden()
    expect(await page.evaluate(() => localStorage.getItem('ps_graph_hints_seen'))).toBe('1')
  })

  test('bottom bar: collapse, Alt+B toggles, the state survives a reload', async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem('ps_graph_hints_seen', '1')
      if (sessionStorage.getItem('ps_e2e_bottom_bar_reload') !== '1') {
        localStorage.removeItem('ps_graph_bottom_bar_collapsed')
      }
    })
    await loadGraphViaFilePicker(page)
    const bar = page.getByTestId('graph-bottom-bar')
    await expect(bar).toHaveAttribute('aria-expanded', 'true')
    await expect(page.getByTestId('graph-status-lens-selector')).toBeVisible()

    await page.getByTestId('graph-bottom-bar-toggle').click()
    await expect(bar).toHaveAttribute('aria-expanded', 'false')
    await expect(page.getByTestId('graph-bottom-bar-expand')).toBeVisible()
    await expect(page.getByTestId('graph-status-lens-selector')).toBeHidden()

    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('Alt+b')
    await expect(bar).toHaveAttribute('aria-expanded', 'true')
    await page.keyboard.press('Alt+b')
    await expect(bar).toHaveAttribute('aria-expanded', 'false')

    // Alt+B typed into a field is the field's business.
    await statusBarCorpusPathInput(page).click()
    await page.keyboard.press('Alt+b')
    await expect(bar).toHaveAttribute('aria-expanded', 'false')

    // Reload and load the graph again. Not via `loadGraphViaFilePicker`: it waits for **Fit**,
    // which lives in the bar and is hidden while the bar is collapsed.
    await page.evaluate(() => sessionStorage.setItem('ps_e2e_bottom_bar_reload', '1'))
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.getByTestId('status-bar-local-file-input').setInputFiles(GI_SAMPLE_FIXTURE)
    await expect(page.locator('.graph-canvas')).toBeVisible()
    await expect(page.getByTestId('graph-bottom-bar')).toHaveAttribute('aria-expanded', 'false')
    await page.getByTestId('graph-bottom-bar-expand').click()
    await expect(page.getByTestId('graph-bottom-bar')).toHaveAttribute('aria-expanded', 'true')
  })
})
