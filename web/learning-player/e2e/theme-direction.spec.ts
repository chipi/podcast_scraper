import { expect, test, type Page } from '@playwright/test'

/**
 * The visual-direction switch, through the real boot path (#1949, #1978 follow-up).
 *
 * ## Why this exists as an e2e rather than only a unit test
 *
 * `src/theme/direction.test.ts` covers the resolution rules against an in-memory store. It cannot
 * see the three things that actually make the feature work: that `main.ts` runs the resolver at
 * boot at all, that the attribute it sets is the one `directions.css` keys on, and that a real
 * `sessionStorage` carries the choice across a full page load. Every one of those is a wiring
 * question, and wiring is exactly what the unit test stubs out.
 *
 * The clearing path is the reason this matters. `?direction=` could not turn the switch off — the
 * stored value survived and was restored on the next navigation — and the only way to observe that
 * bug is to load a page, then load ANOTHER page and look again. A single-page assertion shows a
 * cleared attribute and calls it fixed.
 *
 * Signed-out on purpose: the switch is applied before the app decides anything about a user, and
 * the palette is not a per-account setting. Using the login screen also keeps this independent of
 * fixture content.
 */

/** The `data-direction` attribute on <html>, or null when the default palette is active. */
async function direction(page: Page): Promise<string | null> {
  return page.evaluate(() => document.documentElement.getAttribute('data-direction'))
}

/** What the browser has actually resolved `--lp-canvas` to — proof the stylesheet took effect. */
async function canvas(page: Page): Promise<string> {
  return page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--lp-canvas').trim(),
  )
}

test.describe('visual direction switch', () => {
  test('turns on, survives a full page load, and turns back off', async ({ page }) => {
    // Default: no attribute, and the archive ground.
    await page.goto('/login')
    await page.waitForLoadState('networkidle')
    expect(await direction(page)).toBeNull()
    const defaultCanvas = await canvas(page)
    expect(defaultCanvas.toLowerCase()).toBe('#080d1b')

    // On, by query parameter.
    await page.goto('/login?direction=ember')
    await page.waitForLoadState('networkidle')
    expect(await direction(page)).toBe('ember')

    // The attribute alone proves nothing — assert the STYLESHEET responded. A typo'd direction
    // name would set the attribute and change no colour at all.
    const emberCanvas = await canvas(page)
    expect(emberCanvas.toLowerCase()).toBe('#0e0d10')
    expect(emberCanvas).not.toBe(defaultCanvas)

    // Persists across a full page load with NO parameter — this is the sessionStorage path through
    // main.ts, and it is the half a single-page test cannot see.
    await page.goto('/login')
    await page.waitForLoadState('networkidle')
    expect(await direction(page)).toBe('ember')
    expect((await canvas(page)).toLowerCase()).toBe('#0e0d10')

    // Off, by an explicitly empty value. This is the case that was broken.
    await page.goto('/login?direction=')
    await page.waitForLoadState('networkidle')
    expect(await direction(page)).toBeNull()
    expect((await canvas(page)).toLowerCase()).toBe('#080d1b')

    // And STAYS off on the next load. Before the fix the stored value survived the clear and came
    // back here, so this assertion is the one that would have failed.
    await page.goto('/login')
    await page.waitForLoadState('networkidle')
    expect(await direction(page)).toBeNull()
    expect((await canvas(page)).toLowerCase()).toBe('#080d1b')
  })

  test('switches straight from one direction to another', async ({ page }) => {
    await page.goto('/login?direction=ember')
    await page.waitForLoadState('networkidle')
    expect(await direction(page)).toBe('ember')

    await page.goto('/login?direction=paper')
    await page.waitForLoadState('networkidle')
    expect(await direction(page)).toBe('paper')

    // The replacement must be stored, not merely applied for one load.
    await page.goto('/login')
    await page.waitForLoadState('networkidle')
    expect(await direction(page)).toBe('paper')
  })

  test('no parameter and no stored choice leaves the default untouched', async ({ page }) => {
    // Guards against the switch acquiring a default of its own. Every user who never passes the
    // parameter must get an <html> with no direction attribute at all.
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    expect(await direction(page)).toBeNull()
  })
})
