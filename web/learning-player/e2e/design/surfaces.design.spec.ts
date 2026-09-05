import { expect, test, type Page } from '@playwright/test'
import { expectSignedIn } from '../helpers'

/**
 * Screenshot every surface the redesign touches, at the judged viewport (#1944, #1945).
 *
 * This is the critic loop's INPUT. It is not a test: it asserts only enough to guarantee the
 * screenshot is of a loaded surface rather than a spinner or an empty state. A blank PNG that
 * silently passes would poison the whole exercise — the critic would score an empty screen, and
 * we would act on the number.
 *
 * Output goes to `design-results/<variant>/<surface>.png`, where `<variant>` comes from
 * DESIGN_VARIANT (default `baseline`). So capturing the current app is:
 *
 *   npm run design:shots
 *
 * and a direction is:
 *
 *   DESIGN_VARIANT=warm-brutalist npm run design:shots
 *
 * Keeping variants in sibling folders means the critic can be handed a set with no filenames that
 * reveal intent, and before/after pairs stay trivially available for a PR description.
 */
/**
 * Which direction to shoot, and where the PNGs land.
 *
 * `DESIGN_DIRECTION` names a block in `theme/directions.css`; unset means today's shipping look.
 * The output folder defaults to the direction's own name, so shooting a direction is one variable:
 *
 *   npm run design:shots                        -> design-results/baseline/
 *   DESIGN_DIRECTION=paper npm run design:shots -> design-results/paper/
 *
 * `DESIGN_VARIANT` still overrides the folder on its own, which is what lets the same CSS be shot
 * twice under different names to prove the harness is deterministic.
 */
const DIRECTION = process.env.DESIGN_DIRECTION || ''
const VARIANT = process.env.DESIGN_VARIANT || DIRECTION || 'baseline'
const dir = (name: string) => `design-results/${VARIANT}/${name}.png`

/**
 * Sign in — Library and Profile are auth-gated and render an empty shell signed out.
 *
 * The identity is deliberately CONSTANT across variants, not `design-${VARIANT}` (#1949). Two
 * things went wrong when it varied. Profile prints the account name and address, so a longer
 * variant slug wrapped the line and shifted the whole page down — an 11.8% pixel diff that was
 * pure filename. And each variant got its own fresh account, so whatever state accumulated during
 * one shoot (follows, captures, history) was absent from the next, and surfaces differed for
 * reasons that had nothing to do with the direction being judged.
 *
 * Comparing directions is the entire point of this harness, so the account must be a constant and
 * the stylesheet must be the only variable.
 */
const IDENTITY = 'design-surfaces'

async function signIn(page: Page): Promise<void> {
  // Seeded through sessionStorage rather than `?direction=` on every goto: main.ts reads the key
  // before it paints, so the direction is in force on the FIRST frame of the very first load.
  // Threading a query param through each navigation would leave one unstyled paint at boot, and a
  // screenshot taken near it would judge the wrong stylesheet.
  if (DIRECTION) {
    await page.addInitScript((d) => {
      try {
        sessionStorage.setItem('lp.direction', d)
      } catch {
        /* private-mode storage is not worth failing a screenshot over */
      }
    }, DIRECTION)
  }
  await page.goto(`/api/app/auth/login?as=${IDENTITY}`)
  await expectSignedIn(page)
}

/**
 * Settle the page before shooting.
 *
 * `networkidle` alone is not enough: images decode after the response lands, and a screenshot
 * taken a frame early shows a layout that has not reflowed around them — which reads to a critic
 * as bad spacing rather than a race.
 */
async function settle(page: Page): Promise<void> {
  await page.waitForLoadState('networkidle')
  await page.evaluate(() => {
    // Only images that are ACTUALLY loading. A `loading="lazy"` image below the fold never
    // starts, so it stays `complete === false` forever and its onload never fires — waiting on
    // one hangs until the test times out, which is exactly how Browse and Library first failed
    // here. Same for an <img> with no src yet.
    const pending = Array.from(document.images).filter(
      (i) => !i.complete && i.getAttribute('loading') !== 'lazy' && !!i.currentSrc,
    )
    // Bounded regardless: a decode that stalls should cost a slightly-early screenshot, never the
    // whole capture run.
    const settled = Promise.all(
      pending.map((i) => new Promise((res) => { i.onload = i.onerror = res })),
    )
    return Promise.race([settled, new Promise((res) => setTimeout(res, 3000))])
  })
  // One rAF so any mount transition has committed.
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => r(null))))
}

/**
 * Two framings per surface, and both are needed.
 *
 * `fullPage` shows the whole composition — rail rhythm, section spacing, how the page reads as a
 * single object. But it renders FIXED elements at their scroll position, so the bottom nav lands
 * in the middle of the image. A critic judging that sees a navigation bar floating over the
 * content and marks the layout broken, which is an artifact of the capture, not the design.
 *
 * The viewport shot is what a person actually sees: fixed chrome where it belongs, above the
 * fold. Judge hierarchy and first impression there; judge composition on the full page.
 */
async function shoot(page: Page, name: string): Promise<void> {
  await settle(page)
  await page.screenshot({ path: dir(`${name}-full`), fullPage: true })
  await page.screenshot({ path: dir(`${name}-viewport`), fullPage: false })
}

test('home', async ({ page }) => {
  await signIn(page)
  await page.goto('/')
  // Proof of a populated surface: Home's rails must have rendered something to judge.
  await expect(page.locator('a[href*="/episode/"], a[href*="/podcast/"]').first()).toBeVisible()
  await shoot(page, 'home')
})

test('player', async ({ page }) => {
  await signIn(page)
  await page.goto('/')
  await page.locator('a[href*="/episode/"]').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  // The transport is the surface's centre of gravity — without it this is a screenshot of an
  // error card, and the critic would score the error card.
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeVisible({
    timeout: 30_000,
  })
  await shoot(page, 'player')
})

test('search', async ({ page }) => {
  await signIn(page)
  await page.goto('/search')
  await expect(page.getByRole('searchbox')).toBeVisible()
  await shoot(page, 'search-empty')

  // The populated state is a different composition and is worth judging separately — an empty
  // search page flatters any design.
  await page.getByRole('searchbox').fill('risk')
  await page.getByRole('searchbox').press('Enter')
  await page.waitForLoadState('networkidle')
  await shoot(page, 'search-results')
})

test('browse', async ({ page }) => {
  await signIn(page)
  await page.goto('/browse')
  await expect(page.getByRole('tab').or(page.getByRole('button')).first()).toBeVisible()
  await shoot(page, 'browse')
})

test('library', async ({ page }) => {
  await signIn(page)
  await page.goto('/library')
  await expect(page.getByRole('button', { name: 'Saved', exact: true })).toBeVisible()
  await shoot(page, 'library')
})

test('profile', async ({ page }) => {
  await signIn(page)
  await page.goto('/profile')
  await expect(page.getByTestId('profile-settings-link')).toBeVisible()
  await shoot(page, 'profile')
})
