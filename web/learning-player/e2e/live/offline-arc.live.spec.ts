import { createHmac } from 'node:crypto'
import { expect, test } from '@playwright/test'

/**
 * Post-deploy live smoke for the offline arc (#1905, #1906, #1914, #1925).
 *
 * The existing live suite predates this arc: it covers Home, Search, Browse, Collections, Library
 * and trending, and none of it touches recaps, item-level queue writes, capture idempotency or
 * deep links. So the deploy gate was certifying the OLD surface and waving the new one through.
 * This file closes that, for the half that is testable from a browser.
 *
 * NOT covered here, and it cannot be: downloads and offline playback are device-only. No web smoke
 * can mark an episode for offline, drop the network and prove it still plays. That needs a
 * TestFlight build checked by hand against prod — see `make test-app-ios-journey` for the local
 * equivalent.
 *
 * Auth matches account.live.spec.ts: mint the same HMAC-signed session the app issues, from the
 * prod session secret + a seeded test user. Every write is REVERSIBLE and scoped to that account.
 *
 * Requires (skips cleanly otherwise):
 *   PLAYER_APP_SESSION_SECRET  the prod session-signing secret
 *   PLAYER_SMOKE_USER_ID       a test user seeded in the prod user store
 *   PLAYER_PREVIEW_PASS        the coming-soon gate password
 */
const secret = process.env.PLAYER_APP_SESSION_SECRET || ''
const userId = process.env.PLAYER_SMOKE_USER_ID || ''
const gatePass = process.env.PLAYER_PREVIEW_PASS || ''
const enabled = Boolean(secret && userId && gatePass)

/** Byte-match app_sessions.sign: urlsafe-b64, HMAC-SHA256, compact + sorted-keys JSON. */
function mintSession(): string {
  const b64url = (b: Buffer) =>
    b.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  const json = JSON.stringify({ iat: Math.floor(Date.now() / 1000), user_id: userId })
  const body = b64url(Buffer.from(json, 'utf-8'))
  const sig = b64url(createHmac('sha256', secret).update(body).digest())
  return `${body}.${sig}`
}
const bearer = () => ({ Authorization: `Bearer ${mintSession()}` })

/** First ready episode on the deployed corpus — the specs need a real slug, not a fixture one. */
async function readySlug(request: import('@playwright/test').APIRequestContext): Promise<string> {
  const list = await request.get('/api/app/episodes?page_size=15', { headers: bearer() })
  expect(list.status(), 'episode list should be reachable').toBe(200)
  const items = ((await list.json()).items ?? []) as Array<{ slug: string; status: string }>
  const ep = items.find((e) => e.status === 'ready')
  expect(ep, 'prod needs at least one ready episode for the offline-arc smoke').toBeTruthy()
  return ep!.slug
}

test.describe('offline arc (test account)', () => {
  test.skip(
    !enabled,
    'set PLAYER_APP_SESSION_SECRET + PLAYER_SMOKE_USER_ID + PLAYER_PREVIEW_PASS to run',
  )
  test.beforeEach(async ({ request }) => {
    await request.get('/preview')
  })

  test('every recap window resolves, and an unknown window is rejected', async ({ request }) => {
    // The window contract is the part a deploy can break silently: `ytd` is computed differently
    // from the rolling windows (1 Jan → today, a different length every day), so it is the one
    // most likely to throw on a date the fixture corpus never exercised.
    for (const window of ['week', 'month', 'year', 'ytd'] as const) {
      const resp = await request.get(`/api/app/me/recap?window=${window}`, { headers: bearer() })
      expect(resp.status(), `recap window=${window}`).toBe(200)
      const body = await resp.json()
      expect(body, `recap window=${window} names the window it describes`).toHaveProperty(
        'window',
        window,
      )
    }

    // FastAPI's Literal validation — a typo'd window must be refused, not silently defaulted to
    // "week". A default here would show someone the wrong period and look completely normal.
    const bad = await request.get('/api/app/me/recap?window=fortnight', { headers: bearer() })
    expect(bad.status(), 'an unknown window is a 422, never a silent fallback').toBe(422)
  })

  test('user stats resolve for the signed-in account', async ({ request }) => {
    const resp = await request.get('/api/app/me/stats', { headers: bearer() })
    expect(resp.status()).toBe(200)
    const body = await resp.json()
    // Per-user totals are the user's OWN data, so unlike the public episode stats they are never
    // withheld — a null here means the accrual path broke, not that a floor applied.
    expect(body, 'me/stats returns the user their own totals').toBeTruthy()
  })

  test('item-level queue writes round-trip (add → present → remove → gone)', async ({
    request,
  }) => {
    // The whole point of #1925: the queue is mutated per ITEM, not by PUTting the whole list. A
    // whole-list write is what made an offline queue lose a concurrent change on another device.
    const slug = await readySlug(request)

    const added = await request.post('/api/app/queue/items', {
      headers: bearer(),
      data: { slug },
    })
    expect(added.status(), 'POST /queue/items').toBe(200)
    // QueueResponse.items is list[str] — ordered slugs, not objects.
    expect(
      ((await added.json()).items ?? []) as string[],
      'the queue contains the episode we just added',
    ).toContain(slug)

    const removed = await request.delete(`/api/app/queue/items/${encodeURIComponent(slug)}`, {
      headers: bearer(),
    })
    expect(removed.status(), 'DELETE /queue/items/{slug}').toBe(200)
    expect(
      ((await removed.json()).items ?? []) as string[],
      'the episode is gone after removal — no residue on the test account',
    ).not.toContain(slug)
  })

  test('a highlight posted twice with the same client_id creates ONE row', async ({ request }) => {
    // Capture idempotency (#1925). This is what lets a highlight made offline sit in the outbox
    // and be replayed safely: without it, a retry whose response was lost duplicates the capture.
    // Replaying the same client_id must return the stored row unchanged, not create a second.
    const slug = await readySlug(request)
    const clientId = `smoke-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    // HighlightCreate: `kind` is required; a "moment" is the mark-this-moment capture, anchored
    // by start_ms. There is no free-text field on this model — quote_text is the transcript quote.
    const payload = {
      client_id: clientId,
      episode_slug: slug,
      kind: 'moment' as const,
      start_ms: 12_000,
      quote_text: 'live smoke capture — safe to delete',
    }

    const first = await request.post('/api/app/highlights', { headers: bearer(), data: payload })
    expect(first.status(), 'first POST /highlights is 201 Created').toBe(201)
    const firstBody = await first.json()

    // The replay. A 201 with a NEW id here is the duplicate bug this contract exists to prevent.
    const replay = await request.post('/api/app/highlights', { headers: bearer(), data: payload })
    expect(
      [200, 201],
      'a replayed client_id is accepted, not rejected — the client cannot tell its POST landed',
    ).toContain(replay.status())
    const replayBody = await replay.json()
    expect(
      replayBody.id,
      'replaying the same client_id returns the SAME highlight, not a second one',
    ).toBe(firstBody.id)
    // The server mints `id = client_id or new_id()`, so the client-minted key IS the row id —
    // which is what makes the replay safe without a server-side dedupe table.
    expect(firstBody.id, 'a client-minted id becomes the highlight id').toBe(clientId)

    // And exactly ONE row exists for it, which is the assertion the duplicate bug would break.
    const listed = await request.get('/api/app/highlights', { headers: bearer() })
    expect(listed.status()).toBe(200)
    const matching = (((await listed.json()).items ?? []) as Array<{ id: string }>).filter(
      (h) => h.id === clientId,
    )
    expect(matching.length, 'the replayed capture exists exactly once').toBe(1)

    // Reversible: leave the test account as we found it.
    const deleted = await request.delete(
      `/api/app/highlights/${encodeURIComponent(firstBody.id)}`,
      { headers: bearer() },
    )
    expect(deleted.status(), 'cleanup DELETE /highlights/{id}').toBe(200)
    expect(
      ((await deleted.json()).items ?? []).map((h: { id: string }) => h.id),
      'the smoke leaves no highlight behind',
    ).not.toContain(firstBody.id)
  })

  test('a playback position round-trips with the client timestamp', async ({ request }) => {
    // #1913: the flush carries WHEN the position was reached, so a stale offline write cannot
    // clobber a newer position set on another device.
    const slug = await readySlug(request)
    const put = await request.put(`/api/app/playback/${encodeURIComponent(slug)}`, {
      headers: bearer(),
      data: { position_seconds: 30, finished: false, client_ts: Math.floor(Date.now() / 1000) },
    })
    expect(put.status(), 'PUT /playback/{slug}').toBe(200)

    const got = await request.get(`/api/app/playback/${encodeURIComponent(slug)}`, {
      headers: bearer(),
    })
    expect(got.status()).toBe(200)
    expect((await got.json()).position_seconds, 'the position we just wrote reads back').toBe(30)
  })

  test('a ?t= deep link opens the episode page on the deployed app', async ({ page, context }) => {
    // Deep links (#1925) are how a recap line, a shared quote or an MCP citation gets someone to a
    // MOMENT. Asserted at page level rather than API level because the failure mode is routing:
    // the link resolves, the app boots, and the episode surface renders.
    await context.setHTTPCredentials({
      username: process.env.PLAYER_PREVIEW_USER || 'marko',
      password: gatePass,
    })
    const request = context.request
    await request.get('/preview')
    const slug = await readySlug(request)

    await page.goto(`/episode/${encodeURIComponent(slug)}?t=42`)
    await expect(page).toHaveURL(/t=42/)
    // The transport is what proves the page actually mounted the player rather than an error card.
    await expect(page.locator('audio')).toBeAttached({ timeout: 30_000 })
  })
})
