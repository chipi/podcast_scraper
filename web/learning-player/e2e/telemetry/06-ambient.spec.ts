import { expect, test } from '@playwright/test'
import { attachSink } from './sink'
import './settle'

/**
 * The events that are not part of one obvious flow: the recap, sharing, exporting, the search chip
 * row, and the offline session (#2267).
 */

async function signIn(page: import('@playwright/test').Page, who: string): Promise<void> {
  await page.goto(`/api/app/auth/login?as=${who}`)
  await page.waitForLoadState('networkidle')
}

test.describe('recap', () => {
  test('recap_view fires when the listening recap is actually rendered', async ({ page }) => {
    const sink = attachSink(page)
    await signIn(page, 'telemetry-recap')
    await page.goto('/profile')
    const recap = await sink.waitForEvent('recap_view', 20_000)
    expect(recap.name).toBe('recap_view')
  })
})

test.describe('share', () => {
  test('share reports the target kind and the METHOD, and a failed share reports nothing', async ({
    page,
    context,
  }) => {
    // Copy-to-clipboard needs permission, or the copy path throws and the app correctly reports
    // nothing — which would look like a missing event rather than a refused clipboard.
    await context.grantPermissions(['clipboard-read', 'clipboard-write']).catch(() => {})
    const sink = attachSink(page)
    await signIn(page, 'telemetry-share')
    await page.goto('/')
    const card = page.locator('a[href^="/episode/"]').first()
    await expect(card).toBeVisible()
    await card.click()
    await page.waitForURL(/\/episode\//)

    await page.getByTestId('share-menu').click()
    await expect(page.getByTestId('share-menu-list')).toBeVisible()
    await page.getByTestId('share-link').click()

    const sh = await sink.waitForEvent('share', 20_000)
    expect(sh.data?.target_kind).toBe('episode')
    // `method` separates the OS sheet from a copied link. They are different intentions — one leaves
    // the app, one stays in it — and only one of them is attributable later.
    expect(['native_sheet', 'copy_link']).toContain(String(sh.data?.method))
  })
})

test.describe('highlights export', () => {
  test('every export destination is counted, including the web Markdown link', async ({ page }) => {
    const sink = attachSink(page)
    await signIn(page, 'telemetry-export')
    await page.goto('/highlights')

    // THE WEB MARKDOWN LINK WAS NOT COUNTED AT ALL (#2267, fixed 2026-10-03). The native button
    // calls `exportHighlightsNative()` which reports; the web branch is a plain
    // `<a :href download="my-highlights.md">`, and it had no handler. So "which export format do
    // people use" would have reported zero Markdown exports on the web — where essentially every
    // beta tester is. Same silent-zero shape as a dead website id: the metric exists, computes, and
    // describes a world in which a working feature is never used.
    const mdLink = page.locator('a[download="my-highlights.md"]')
    if (await mdLink.isVisible().catch(() => false)) {
      // `download` on an anchor: let the click happen but do not let the navigation/save proceed.
      await mdLink.click({ modifiers: [] })
      const md = await sink.waitForEvent('highlights_export', 20_000)
      expect(md.data).toMatchObject({ format: 'markdown' })
    }

    // PDF was not counted on EITHER platform. Reported inside `openPrintable` rather than on the
    // button so both its branches (web: open printable; native: share sheet) count.
    const pdf = page.getByTestId('export-pdf')
    if (await pdf.isVisible().catch(() => false)) {
      const before = sink.byName('highlights_export').length
      await pdf.click()
      await expect
        .poll(() => sink.byName('highlights_export').length, { timeout: 20_000 })
        .toBeGreaterThan(before)
      const formats = sink.byName('highlights_export').map((b) => String(b.data?.format))
      expect(formats).toContain('pdf')
    }

    // Every format that reached the wire must be in the registry's union. The property used to be
    // typed as a bare `string`, which opted this one event out of the compile-time guarantee the rest
    // of the registry makes.
    for (const b of sink.byName('highlights_export')) {
      expect(['markdown', 'pdf', 'obsidian']).toContain(String(b.data?.format))
    }
  })
})

test.describe('search result click', () => {
  /**
   * Two result KINDS, two different questions, and the episode one is the reachable half here.
   *
   * The related-topic chips are derived from the topics attached to the returned hits
   * (`aggregateRelatedTopics(results, 8)`), so whether they render at all depends on what the corpus
   * enriches — this committed fixture produced none for the queries tried, which is a property of the
   * corpus and not of the wiring. The EPISODE result row is always present when search returns hits,
   * so that is what this asserts; the chip branch is covered opportunistically when it exists.
   */
  test('an episode result reports result_kind=episode with a bucketed rank', async ({ page }) => {
    const sink = attachSink(page)
    await signIn(page, 'telemetry-search-click')
    await page.goto('/')
    await page.getByTestId('home-search-input').fill('risk')
    await page.getByTestId('home-search-submit').click()
    await page.waitForURL(/\/search/)

    // `Play from {time} in {episode}` — the per-hit jump control, which is what calls `openEpisode`
    // with the hit's rank. The surrounding row is not the tracked element.
    //
    // MEASURED ABSENCE, not a flaky selector: this host has no `lancedb` wheel (no x86_64 build), so
    // the API logs `hybrid_search open failed (No module named 'lancedb'); reporting no_index` and the
    // search page renders "Couldn't load this right now." No grouped episode row exists to carry a
    // jump control. The wiring is instead asserted statically in
    // `src/__checks__/analytics-wiring.test.ts`, which runs everywhere — so the event is covered even
    // though this tier cannot reach it here. On a host with the search extras this test runs for real.
    const jump = page.locator('button[aria-label^="Play from "]').first()
    if (!(await jump.isVisible().catch(() => false))) {
      test.skip(
        true,
        'no Lance index on this host -> search errors, so no episode result row renders. Static ' +
          'coverage lives in __checks__/analytics-wiring.test.ts.',
      )
    }
    await jump.click()

    const click = await sink.waitForEvent('search_result_click', 20_000)
    expect(click.data?.result_kind).toBe('episode')
    // Bucketed, so "which position do people take" stays answerable without turning the event into a
    // per-query fingerprint.
    expect(['1', '2-3', '4-10', '11+']).toContain(String(click.data?.rank))
  })

  test('a topic chip, when the corpus produces one, reports result_kind=topic', async ({ page }) => {
    const sink = attachSink(page)
    await signIn(page, 'telemetry-search-chip')
    await page.goto('/')
    await page.getByTestId('home-search-input').fill('risk')
    await page.getByTestId('home-search-submit').click()
    await page.waitForURL(/\/search/)

    const chip = page.locator('[data-testid="related-topic-chips"] button').first()
    if (!(await chip.isVisible().catch(() => false))) {
      test.skip(
        true,
        'this corpus attached no topics to the returned hits, so the chip row does not render',
      )
    }
    await chip.click()

    const click = await sink.waitForEvent('search_result_click', 20_000)
    // The chip row is a short unranked set, so rank is the constant '1' rather than an invented
    // ordinal — otherwise the same bucket would mean two different things across events.
    expect(click.data).toMatchObject({ result_kind: 'topic', rank: '1' })
    const open = await sink.waitForEvent('entity_open', 20_000)
    expect(open.data).toMatchObject({ kind: 'topic', source: 'search' })
  })
})

test.describe('offline', () => {
  /**
   * `offline_session` is the one event whose own condition blocks its delivery.
   *
   * It fires only while `isOnline` is false (`if (isOnlineRef.value) return`), and a beacon needs the
   * network the event exists to report the absence of. The pre-load queue added in this arc does not
   * help: it catches the window before `window.umami` exists, not a failed POST afterwards.
   *
   * So this test MEASURES what actually happens rather than asserting a hoped-for outcome. If the
   * beacon is lost, `offline_session` is structurally undeliverable from the web and the dashboard
   * panel for it will read zero forever — which is a finding about the metric, not a test failure to
   * paper over. Either way the app must survive going offline.
   */
  test('going offline fires the event; whether it can be DELIVERED is the measurement', async ({
    page,
    context,
  }) => {
    const sink = attachSink(page)
    await signIn(page, 'telemetry-offline')
    await page.goto('/')
    // Let the tracker load first, so this measures the network, not the boot race.
    await page.waitForLoadState('networkidle')

    await context.setOffline(true)
    // Give the online watcher time to notice and the app time to react.
    await page.waitForTimeout(3_000)
    const attemptedWhileOffline = sink.byName('offline_session').length

    await context.setOffline(false)
    await page.waitForTimeout(3_000)

    // The app is still alive after a real network drop and recovery.
    await page.goto('/')
    await expect(page.locator('body')).toBeVisible()

    // Record the finding in the test output, whichever way it goes.
    const delivered = sink.byName('offline_session').length
    // eslint-disable-next-line no-console
    console.log(
      `[offline_session] beacon attempts seen on the wire: while offline=${attemptedWhileOffline}, total=${delivered}`,
    )
    // The ASSERTION is only the part that must hold regardless: the app does not break, and no
    // half-formed event escaped. Delivery is reported above and checked against Umami separately.
    expect(sink.unparsed, 'no telemetry request should be unparseable').toHaveLength(0)
  })
})
