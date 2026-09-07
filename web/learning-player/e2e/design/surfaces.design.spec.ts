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

/**
 * Move the playhead without actually playing — sets `currentTime` and fires the `timeupdate` the
 * player store listens for, the same recipe `recap-and-deep-links.spec.ts` uses. A real `.play()`
 * would work too but is slower and non-deterministic for a screenshot.
 */
async function seekTo(page: Page, seconds: number): Promise<void> {
  await page.locator('audio').waitFor({ state: 'attached' })
  await page.evaluate((t) => {
    const el = document.querySelector('audio') as HTMLAudioElement | null
    if (!el) throw new Error('no audio element')
    el.currentTime = t
    el.dispatchEvent(new Event('timeupdate'))
  }, seconds)
}

/**
 * The artwork's Zone D intelligence band (#Zone-D rewrite) — the surface this exploration is
 * about, so it gets its own two shots rather than relying on whatever moment `player` above
 * happens to land on at t=0 (always the rest state, since nothing has played yet).
 *
 * Episode p05_e03 ("The Bessent Tape") has three timed, non-degenerate insights packed into its
 * first 36 seconds and nothing after — real corpus data, not a fixture built for this shot.
 */
test('player zoneD — live insight', async ({ page }) => {
  await signIn(page)
  await page.goto('/podcast/p05')
  await page.getByText('The Bessent Tape').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeVisible({
    timeout: 30_000,
  })
  // 3s sits inside the first insight's [0, 6s] supporting-quote window.
  await seekTo(page, 3)
  await expect(page.getByTestId('player-zone-d-live')).toBeVisible()
  await shoot(page, 'zoneD-live')
})

test('player zoneD — rest (no insight active)', async ({ page }) => {
  await signIn(page)
  await page.goto('/podcast/p05')
  await page.getByText('The Bessent Tape').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeVisible({
    timeout: 30_000,
  })
  // Past the last insight's window (ends at 36s) AND its linger (+4s) — nothing is "surfacing
  // now", which is exactly the state that used to fall back to showing the FIRST insight instead
  // of nothing. This is the shot that proves that bug is gone.
  await seekTo(page, 50)
  await expect(page.getByTestId('player-zone-d-rest')).toBeVisible()
  await shoot(page, 'zoneD-rest')
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
  // Library's tabs are `role="tab"` since #1594 item 7 — they previously carried NO role at
  // all, which is why `getByRole('button')` matched them.
  await expect(page.getByRole('tab', { name: 'Saved', exact: true })).toBeVisible()
  await shoot(page, 'library')
})

test('profile', async ({ page }) => {
  await signIn(page)
  await page.goto('/profile')
  await expect(page.getByTestId('profile-settings-link')).toBeVisible()
  await shoot(page, 'profile')
})


/**
 * The panel at the LONGEST insight that exists in the corpus.
 *
 * Measured across 355 insights in the committed corpus: median 105 chars, p90 197, max 266 — which
 * renders around 9 lines. That is where the panel pushes furthest up the artwork and where its
 * scrim comes nearest the top toolbar, so it is the case most likely to break. Shooting only a
 * median-length insight leaves it unproven, which is exactly how a fixed-height scrim assumption
 * survived an earlier round.
 *
 * The long text is injected by INTERCEPTING THE INSIGHTS RESPONSE, not by editing the DOM. A first
 * attempt replaced "the largest text node" in the rendered panel and silently clobbered a
 * container — wiping the attribution line and the NEXT row, producing a screenshot that looked
 * like broken behaviour and was really a broken test. Mocking the data lets the component compose
 * itself, which is the only version worth judging.
 *
 * The string is the genuine longest insight in the corpus (p05_e04.gi.json), not invented filler.
 */
test('player zoneD — longest insight in the corpus', async ({ page }) => {
  const LONGEST =
    'Welcome back to Long Horizon Notes. Today is a debate — one question, two people who genuinely ' +
    'disagree, and a host who is going to keep both of them honest about what they actually believe.'

  await page.route(/\/api\/app\/episodes\/[^/]+\/insights/, async (route) => {
    const res = await route.fetch()
    const body = await res.json()
    // Lengthen the first insight in place; every other field, including its quote windows, is the
    // real fixture's, so the timing logic still behaves exactly as it does in production.
    if (Array.isArray(body.insights) && body.insights[0]) body.insights[0].text = LONGEST
    await route.fulfill({ response: res, json: body })
  })

  await signIn(page)
  await page.goto('/podcast/p05')
  await page.getByText('The Bessent Tape').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeVisible({
    timeout: 30_000,
  })
  await seekTo(page, 3)
  await expect(page.getByTestId('player-zone-d-live')).toBeVisible()
  await shoot(page, 'zoneD-longest')
})
