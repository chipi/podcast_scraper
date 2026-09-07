import { expect, test } from '@playwright/test'
import { addSessionCookie, bearer, canMintSession } from './session'

/**
 * Post-deploy live smoke for RFC-103 R2 trending — the window selector + denoised, corpus-anchored
 * momentum — against the DEPLOYED closelistening.app (playwright.live.config.ts).
 *
 * DATA-AGNOSTIC by design: prod's trending topics change constantly, so this asserts the SHAPE and
 * BEHAVIOUR (the `window` contract, the denoise invariants, the selector wiring) and never a
 * specific topic. It directly guards the regression R2 shipped to fix — a uniform velocity across
 * every entity (the "everything is 2.6×" bug) — and proves the R2 endpoint is actually deployed
 * (the pre-R2 API silently ignores `window` and never 422s on a bad one).
 *
 * The API checks ride the same preview basic-auth as the UI (httpCredentials in the live config),
 * so they skip cleanly when PLAYER_PREVIEW_PASS is unset.
 */
const gated = Boolean(process.env.PLAYER_PREVIEW_PASS)

test.describe('trending (RFC-103 R2)', () => {
  test.skip(
    !gated || !canMintSession,
    'needs PLAYER_PREVIEW_PASS + PLAYER_APP_SESSION_SECRET + PLAYER_SMOKE_USER_ID',
  )

  // Two doors since RFC-120 (#1940): the coming-soon gate AND an app session. Priming /preview puts
  // `cl_preview` in the request jar (an explicit `Authorization: Bearer` would otherwise override
  // the Basic that clears the gate), and the session cookie signs the PAGE in — without it every
  // navigation below lands on /welcome instead of the app.
  test.beforeEach(async ({ request, page, baseURL }) => {
    await request.get('/preview')
    await addSessionCookie(page.context(), baseURL || 'https://closelistening.app')
  })

  test('the trending API honours the window contract and is denoised', async ({ request }) => {
    const resp = await request.get('/api/app/trending?kind=topic&window=3m&limit=20', {
      headers: bearer(),
    })
    expect(resp.status()).toBe(200)
    const body = await resp.json()
    expect(body.window).toBe('3m')
    const items: Array<{ velocity: number; total: number; window?: string }> = body.items ?? []
    // Prod ships real trending, so with >1 item both R2 invariants must hold:
    if (items.length > 1) {
      // (a) NOT the uniform-velocity regression — the list must carry >1 distinct velocity.
      const distinct = new Set(items.map((i) => i.velocity))
      expect(
        distinct.size,
        'every velocity is identical → the uniform-2.6× regression is back'
      ).toBeGreaterThan(1)
      // (b) the min_total INCLUSION floor — no one-off singletons leak into the list.
      for (const i of items) expect(i.total).toBeGreaterThanOrEqual(3)
      // (c) each row is stamped with the window it was ranked under.
      expect(items.every((i) => i.window === '3m')).toBe(true)
    }
  })

  test('all four windows resolve and an unknown window 422s', async ({ request }) => {
    for (const w of ['1m', '3m', '6m', '1y']) {
      const r = await request.get(`/api/app/trending?kind=topic&window=${w}`, {
        headers: bearer(),
      })
      expect(r.status(), `window=${w} must resolve`).toBe(200)
    }
    // Proves the R2 endpoint is live: the pre-R2 API ignored unknown query params and returned 200.
    const bad = await request.get('/api/app/trending?kind=topic&window=nope', {
      headers: bearer(),
    })
    expect(bad.status(), 'unknown window must 422 (guards that the R2 endpoint is deployed)').toBe(
      422
    )
  })

  test('Browse Topics shows the window selector, defaults to 3M, and switches', async ({
    page,
  }) => {
    await page.goto('/preview')
    await page.goto('/browse?tab=topics')
    // The hub keep-alives every tab panel (v-show), so BOTH the topics and people panels carry a
    // trend-window-tabs — scope to the topics panel or the locator is ambiguous (strict-mode).
    const panel = page.getByTestId('topic-browse-view')
    await expect(panel.getByTestId('trend-window-tabs')).toBeVisible()
    await expect(panel.getByTestId('trend-window-3m')).toHaveAttribute('aria-selected', 'true')
    await panel.getByTestId('trend-window-6m').click()
    await expect(panel.getByTestId('trend-window-6m')).toHaveAttribute('aria-selected', 'true')
    await expect(panel.getByTestId('trend-window-3m')).toHaveAttribute('aria-selected', 'false')
  })
})
