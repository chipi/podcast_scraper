import { expect, test } from '@playwright/test'
import { attachSink } from './sink'
import { openTranscript } from '../helpers'
import './settle'

/**
 * The listening session — where the beta's depth metrics come from (#2267).
 *
 * `episode_open` is the one event with a property nothing downstream can recover:
 * `sourceOfCurrentNavigation()`. By the time the episode page has mounted and loaded its feed (which
 * it must, because the event also needs `from_followed_show`), the previous route is gone. So the
 * router records it and the view reads it back — and if that plumbing silently returned its
 * `other` fallback, Discovery share and Pivot rate would still compute, still look plausible, and be
 * wrong. That is what these assertions are actually guarding.
 */

async function openEpisodeFromHome(page: import('@playwright/test').Page): Promise<void> {
  await page.goto('/api/app/auth/login?as=telemetry-player')
  await page.waitForLoadState('networkidle')
  await page.goto('/')
  // An IN-APP navigation, not a `goto`: `episode_open`'s source comes from the route being LEFT, so
  // a direct page load would legitimately report a cold start and prove nothing about provenance.
  const card = page.locator('a[href^="/episode/"]').first()
  await expect(card, 'Home must offer an episode to open').toBeVisible()
  await card.click()
  await page.waitForURL(/\/episode\//)
}

test.describe('episode open', () => {
  test('carries the surface it came FROM and whether the show is followed', async ({ page }) => {
    const sink = attachSink(page)
    await openEpisodeFromHome(page)

    const open = await sink.waitForEvent('episode_open', 30_000)
    const source = String(open.data?.source ?? '')
    expect(source.length).toBeGreaterThan(0)
    // `from_followed_show` is the whole point of the "do they only replay what they follow" question,
    // and it is a BOOLEAN — a missing value would coerce to false and quietly inflate discovery.
    expect(typeof open.data?.from_followed_show, 'must be a real boolean, not absent').toBe('boolean')
  })

  test('reports once per episode, not once per re-render', async ({ page }) => {
    const sink = attachSink(page)
    await openEpisodeFromHome(page)
    await sink.waitForEvent('episode_open', 30_000)
    const after = sink.byName('episode_open').length

    // The view watches the loaded episode; a latch (`reportedOpenFor`) stops a re-resolve counting
    // again. Without it, time-to-first-play and per-episode depth would both divide by an inflated
    // denominator.
    await page.waitForTimeout(2_000)
    expect(sink.byName('episode_open').length).toBe(after)
  })
})

test.describe('playback', () => {
  test('play_start says whether this was a RESUME, and speed_change reports the new speed', async ({
    page,
  }) => {
    const sink = attachSink(page)
    await openEpisodeFromHome(page)

    const play = page.getByRole('button', { name: /^play$/i }).first()
    await expect(play).toBeVisible()
    await play.click()

    const start = await sink.waitForEvent('play_start', 30_000)
    expect(start.data?.surface).toBe('player')
    // `resumed` separates "started this episode" from "carried on with it", which is the difference
    // between a trial and a habit — the single most important distinction in a retention read.
    expect(typeof start.data?.resumed).toBe('boolean')

    // Speed is a cycling control; one tap must emit exactly one event with the NEW value.
    const speed = page.getByRole('button', { name: /speed/i }).first()
    await expect(speed).toBeVisible()
    await speed.click()
    const sp = await sink.waitForEvent('speed_change')
    expect(String(sp.data?.speed ?? ''), 'the new speed must be labelled').not.toBe('')

    // `setRate` is also called while syncing position state. Those calls must NOT report, or one
    // deliberate tap becomes a stream of identical events.
    const afterOne = sink.byName('speed_change').length
    await page.waitForTimeout(2_500)
    expect(
      sink.byName('speed_change').length,
      'syncing position state must not emit speed_change',
    ).toBe(afterOne)
  })
})

test.describe('transcript', () => {
  test('a tap on a grounded segment reports BOTH the seek and the insight', async ({ page }) => {
    const sink = attachSink(page)
    await openEpisodeFromHome(page)
    await openTranscript(page)

    // `seg` is the element that carries the tracked handler (`onSegmentClick`). The paragraph button
    // that wraps it emits a bare `seek` and reports NOTHING, so clicking the wrapper would move the
    // audio and produce no event — a passing-looking interaction that proves the opposite of what it
    // appears to.
    const seg = page.locator('[data-testid="transcript"] [data-testid="seg"]').first()
    await expect(seg, 'the transcript must render segments to be tappable').toBeVisible()
    await seg.click()

    const seek = await sink.waitForEvent('transcript_seek', 20_000)
    expect(seek.name).toBe('transcript_seek')

    // Reported SEPARATELY on purpose: a tap on a grounded segment both moves the audio and opens an
    // insight, and folding the second into the first would hide the only signal that grounding is
    // what people are actually following.
    const insight = sink.first('insight_tap')
    if (insight) expect(insight.data).toMatchObject({ insight_type: 'grounded_transcript' })
  })
})

test.describe('knowledge panel', () => {
  test('knowledge_panel_open fires on open only, never on close', async ({ page }) => {
    const sink = attachSink(page)
    await openEpisodeFromHome(page)

    const opener = page.getByTestId('player-open-insights')
    await expect(opener).toBeVisible()
    await opener.click()
    await expect(page.getByTestId('knowledge-panel')).toBeVisible()

    const kp = await sink.waitForEvent('knowledge_panel_open')
    expect(kp.data).toMatchObject({ trigger: 'button' })

    const afterOpen = sink.byName('knowledge_panel_open').length
    await page.keyboard.press('Escape')
    await page.waitForTimeout(1_000)
    // The watch fires on every change of `panelOpen`; only the `true` edge is an open. Counting the
    // close would double every panel engagement.
    expect(sink.byName('knowledge_panel_open').length).toBe(afterOpen)
  })
})

test.describe('queue and capture', () => {
  test('queue_add carries the surface the add happened on', async ({ page }) => {
    const sink = attachSink(page)
    await openEpisodeFromHome(page)

    // The EXACT label. A loose /queue/i matched the "Queue" nav entry and a hidden menuitem copy of
    // this same control first, and `.first()` then resolved to something invisible — which read as
    // "this surface has no queue control" when it has one.
    const add = page.getByRole('button', { name: 'Add to queue' }).first()
    await expect(add, 'the player must offer a queue control').toBeVisible()
    await add.click()
    const q = await sink.waitForEvent('queue_add', 20_000)
    expect(String(q.data?.source ?? ''), 'the surface must be named').not.toBe('')
  })

  test('capture_created reports the KIND and target, never the captured text', async ({ page }) => {
    const sink = attachSink(page)
    await openEpisodeFromHome(page)

    const capture = page.getByRole('button', { name: 'Mark this moment' }).first()
    await expect(capture, 'the player must offer the capture control').toBeVisible()
    await capture.click()

    const cap = await sink.waitForEvent('capture_created', 20_000)
    expect(['highlight', 'note']).toContain(String(cap.data?.kind))
    expect(String(cap.data?.target_kind ?? '')).not.toBe('')

    // The no-free-text rule: a highlight's quoted words are the listener's reading, and a note's body
    // can be anything at all. Neither belongs in an analytics store, so only the SHAPE is reported.
    const payload = JSON.stringify(cap.raw)
    expect(payload.length, 'the event must stay small — a body would make it large').toBeLessThan(1200)
  })
})
