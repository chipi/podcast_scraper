import { expect, test } from '@playwright/test'
import { attachSink } from './sink'
import './settle'

/**
 * Discovery — the half of the spec that answers its actual question (#2267).
 *
 * The beta exists to find out whether people MOVE across the corpus (topic → person → another show's
 * episode) or only replay shows they already follow. Every metric that answers it — Discovery share,
 * Pivot rate, cross-show hop — is computed from `entity_open`'s `source` and `presentation`. So
 * these assertions are mostly about those two properties being RIGHT, not about events existing:
 * a `source` that silently falls back to `other` leaves the headline metric computable, plausible,
 * and wrong.
 */

test.describe('browse', () => {
  test('browse_tab_view reports the tab CHOSEN, not the default landed on', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto('/api/app/auth/login?as=telemetry-browse')
    await page.waitForLoadState('networkidle')
    await page.goto('/browse')
    await expect(page.getByTestId('browse-view')).toBeVisible()

    // Nothing yet: arriving at the hub is not choosing a tab. If it were, `episodes` would always
    // lead simply because it renders first — a ranking that measures tab ORDER, not preference.
    const beforeAny = sink.byName('browse_tab_view').length

    await page.getByRole('tab', { name: /shows/i }).click()
    await expect
      .poll(() => sink.byName('browse_tab_view').length, { timeout: 10_000 })
      .toBeGreaterThan(beforeAny)
    const first = sink.byName('browse_tab_view').slice(-1)[0]
    expect(first.data?.tab).toBe('shows')
  })

  test('entity_open from Browse reports presentation=page and source=browse', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto('/api/app/auth/login?as=telemetry-browse-entity')
    await page.waitForLoadState('networkidle')
    // Browse has exactly two tabs — Episodes and Shows. There is no topics tab, and `?tab=topics`
    // silently falls back to Episodes, which is why reaching for a topic row there skipped: the
    // selector was describing a surface that does not exist. Browse's `entity_open` comes from the
    // TRENDS explorer above the tabs (`DiscoveryExplorer @open="onEntityOpen"`), and `?trends=topic`
    // is the query that selects its kind — deliberately not `?tab=`, which drives the tabs instead.
    await page.goto('/browse?trends=topic')
    await expect(page.getByTestId('browse-view')).toBeVisible()

    // Mine is the default and this account has no world yet: read everyone's.
    await page.getByTestId('home-trending-scope-everyone').click()
    const row = page.locator('#trends [data-testid="discovery-row"]').first()
    await expect(
      row,
      'the trends explorer must render a row for Browse to be able to open an entity',
    ).toBeVisible()
    await row.click()

    const open = await sink.waitForEvent('entity_open')
    expect(open.data?.presentation, 'Browse opens a full page, not an overlay card').toBe('page')
    expect(open.data?.source).toBe('browse')
    // The rank question `home_rail_click` asked on Home, asked on Discover (operator 2026-10-07).
    const click = await sink.waitForEvent('trends_row_click')
    expect(click.data?.kind).toBe('topic')
    expect(['1', '2-3', '4-10', '11+'], 'rank is bucketed').toContain(String(click.data?.rank))
  })
})

test.describe('follow', () => {
  test('follow then unfollow are distinct events carrying the surface they happened on', async ({
    page,
  }) => {
    const sink = attachSink(page)
    await page.goto('/api/app/auth/login?as=telemetry-follow')
    await page.waitForLoadState('networkidle')
    await page.goto('/podcast/p05')

    const toggle = page.getByTestId('follow-show')
    await expect(toggle).toBeVisible()
    const pressedBefore = await toggle.getAttribute('aria-pressed')

    await toggle.click()
    const firstName = pressedBefore === 'true' ? 'unfollow' : 'follow'
    const first = await sink.waitForEvent(firstName)
    expect(first.data).toMatchObject({ kind: 'show' })
    // The source comes from the route, because a follow is tappable from Home, an entity card, an
    // entity page and the library — and the spec asks which surface converts.
    expect(first.data?.source).toBe('entity_page')

    // Toggle back: the OPPOSITE event, not a second copy of the first. Reported on intent next to
    // the optimistic state change, so a follow made offline still counts — it is a real follow.
    await expect(toggle).toHaveAttribute('aria-pressed', pressedBefore === 'true' ? 'false' : 'true')
    await toggle.click()
    const secondName = firstName === 'follow' ? 'unfollow' : 'follow'
    await sink.waitForEvent(secondName)
  })
})

test.describe('search', () => {
  test('search_submitted carries scope and a bucketed result count', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto('/api/app/auth/login?as=telemetry-search')
    await page.waitForLoadState('networkidle')
    await page.goto('/')

    await page.getByTestId('home-search-input').fill('risk management')
    await page.getByTestId('home-search-submit').click()
    await page.waitForURL(/\/search/)

    const sub = await sink.waitForEvent('search_submitted', 30_000)
    // Reported AFTER the response, because the count is the point: a submit-time event cannot carry
    // it, and `search_submitted` without a result count cannot tell a search that worked from one
    // that found nothing.
    expect(['corpus', 'recall']).toContain(String(sub.data?.scope))
    expect(['0', '1', '2-5', '6-20', '21+']).toContain(String(sub.data?.results))
  })

  /**
   * These two describe FAILURE states, and the only honest way to cover them is to cause the
   * failure.
   *
   * The rest of this tier refuses mocks on principle — a mocked happy path is how a website id that
   * did not exist passed as wired. This is the opposite situation: the committed corpus returns
   * 6-20 low-scoring hits for literal nonsense (measured: `score: 0.0227` for
   * "zzzz no such subject zzzz"), so no query reaches either branch, and the server cannot be asked
   * to fail on demand. Forcing the RESPONSE is therefore not substituting for the thing under test
   * — the thing under test is what the app reports when the server fails or finds nothing, and the
   * app's real code path runs either way.
   *
   * The distinction matters because "we broke" and "the corpus does not contain this" are different
   * findings. A dashboard that merges them cannot tell a bug from a content gap, and during a beta
   * that is the difference between fixing search and buying more episodes.
   */
  test('a server failure reports error_shown, not an empty state', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto('/api/app/auth/login?as=telemetry-search-error')
    await page.waitForLoadState('networkidle')
    await page.route('**/api/app/search**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ query: 'x', results: [], error: 'embed_failed' }),
      }),
    )
    await page.goto('/')
    await page.getByTestId('home-search-input').fill('risk')
    await page.getByTestId('home-search-submit').click()
    await page.waitForURL(/\/search/)

    const err = await sink.waitForEvent('error_shown', 30_000)
    expect(err.data).toMatchObject({ surface: 'search', kind: 'server' })
    expect(
      sink.byName('empty_state_shown'),
      'a server error must never be reported as "not in corpus" — that blames the content for a bug',
    ).toHaveLength(0)
  })

  test('a genuinely empty result reports empty_state_shown, not an error', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto('/api/app/auth/login?as=telemetry-search-noresults')
    await page.waitForLoadState('networkidle')
    await page.route('**/api/app/search**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ query: 'x', results: [] }),
      }),
    )
    await page.goto('/')
    await page.getByTestId('home-search-input').fill('risk')
    await page.getByTestId('home-search-submit').click()
    await page.waitForURL(/\/search/)

    const empty = await sink.waitForEvent('empty_state_shown', 30_000)
    expect(empty.data).toMatchObject({ surface: 'search', reason: 'not_in_corpus' })
    expect(
      sink.byName('error_shown'),
      'an honest empty result must not inflate the error count',
    ).toHaveLength(0)
  })
})
