import { expect, test } from '@playwright/test'
import { bearer, canMintSession } from './session'

/**
 * Post-deploy live smoke for the k-anonymity floor on cross-user reach (#1923).
 *
 * This is a PRIVACY control, and it is the one surface where a regression is silent: if the floor
 * stops applying, the endpoint keeps returning 200 with a perfectly well-formed body — it just
 * starts telling anyone who asks that exactly two people listened to a given episode. Nothing goes
 * red. Nobody gets an error. The data is simply out. That is why it gets its own live spec rather
 * than a line inside the general smoke.
 *
 * `GET /api/app/episodes/{slug}/stats` used to be PUBLIC, and this comment used to say the floor
 * existed *because* it was. RFC-120 (#1940) put `Depends(get_current_user)` on it along with the
 * rest of the content API, so that premise is gone — but the control is not. The response is
 * cross-user aggregate reach: an exact count still re-identifies when the user base is small, and
 * every signed-in user can read it about every episode. Being behind a login narrows who can ask;
 * it does not make the answer safe to give. `listeners` and `opens` are nulled below
 * `K_ANONYMITY_MIN_LISTENERS` (5). Null means "not enough people", never "nobody".
 *
 * That auth change is also what broke this spec: it primed the coming-soon gate and never signed
 * in, so it got 401 where it asserts 200 and failed the post-deploy smoke on two consecutive prod
 * deploys. It authenticates as the seeded smoke account now, like `account.live.spec.ts`.
 *
 * The assertions are deliberately DATA-INDEPENDENT. A live smoke cannot know how many people used
 * prod today, so asserting a number would either be wrong tomorrow or be so loose it proves
 * nothing. Instead it asserts the two invariants that must hold for any data at all:
 *
 *   1. a disclosed `listeners` count is never below the floor
 *   2. `opens` is never disclosed while `listeners` is withheld
 *
 * (2) matters because `opens` is the easier leak to miss: it is a different counter, computed on a
 * different path, and withholding one while publishing the other still re-identifies — a single
 * listener with eleven opens is as identifying as the listener count itself.
 */

// Sample breadth, not depth: enough episodes that a floor regression is very likely to hit at
// least one with real traffic, few enough that the smoke stays quick on a cold deploy.
const SAMPLE = 12
const FLOOR = 5

test.describe('k-anonymity floor on cross-user episode stats', () => {
  // Needs the gate password AND a mintable session — the endpoints are auth-required now, so the
  // gate alone is no longer enough to reach them.
  test.skip(
    !canMintSession,
    'set PLAYER_PREVIEW_PASS + PLAYER_APP_SESSION_SECRET + PLAYER_SMOKE_USER_ID to run',
  )

  // The coming-soon gate fronts everything; prime it so the request jar carries cl_preview.
  test.beforeEach(async ({ request }) => {
    await request.get('/preview')
  })

  test('a disclosed listener count is never below the floor, and opens never leak past it', async ({
    request,
  }) => {
    const list = await request.get(`/api/app/episodes?page_size=${SAMPLE}`, { headers: bearer() })
    expect(list.status(), 'episode list should be reachable behind the gate').toBe(200)
    const items = ((await list.json()).items ?? []) as Array<{ slug: string; status: string }>
    const slugs = items.filter((e) => e.status === 'ready').map((e) => e.slug)
    expect(slugs.length, 'need at least one ready episode to assert against').toBeGreaterThan(0)

    const violations: string[] = []
    let disclosed = 0
    let withheld = 0

    for (const slug of slugs) {
      const resp = await request.get(`/api/app/episodes/${encodeURIComponent(slug)}/stats`, {
        headers: bearer(),
      })
      expect(resp.status(), `stats for ${slug}`).toBe(200)
      const body = (await resp.json()) as {
        listeners: number | null
        opens: number | null
      }

      if (body.listeners === null) {
        withheld += 1
        // Invariant 2 — the counter that is easiest to forget.
        if (body.opens !== null) {
          violations.push(
            `${slug}: listeners withheld but opens disclosed as ${body.opens} — a single ` +
              `listener with a distinctive open count re-identifies just as well`,
          )
        }
      } else {
        disclosed += 1
        // Invariant 1 — the floor itself.
        if (body.listeners < FLOOR) {
          violations.push(
            `${slug}: listeners=${body.listeners} disclosed below the floor of ${FLOOR}`,
          )
        }
      }
    }

    expect(violations, violations.join('\n  ')).toEqual([])

    // Report what the run actually observed. A smoke where EVERY episode is withheld cannot
    // distinguish "the floor works" from "the endpoint returns null unconditionally", so say so
    // rather than reporting a pass that proves less than it appears to.
    console.log(
      `[k-anonymity] ${slugs.length} episodes: ${disclosed} disclosed, ${withheld} withheld ` +
        `(floor=${FLOOR})`,
    )
    if (disclosed === 0) {
      console.log(
        '[k-anonymity] NOTE: nothing was above the floor on this run, so invariant 1 was not ' +
          'exercised. This passes, but it does not prove the floor releases counts correctly.',
      )
    }
  })

  test('the endpoint stays public and well-formed (the floor is not enforced by 4xx)', async ({
    request,
  }) => {
    // A tempting "fix" for a floor regression is to start refusing the request. That would break
    // every caller and is not what the contract says: the endpoint answers 200 and nulls the
    // fields. Assert the shape so a change of mechanism is caught here rather than in the client.
    const list = await request.get('/api/app/episodes?page_size=1', { headers: bearer() })
    const first = ((await list.json()).items ?? [])[0] as { slug: string } | undefined
    expect(first, 'need an episode to probe').toBeTruthy()

    const resp = await request.get(`/api/app/episodes/${encodeURIComponent(first!.slug)}/stats`, {
      headers: bearer(),
    })
    expect(resp.status()).toBe(200)
    const body = await resp.json()
    expect(body).toHaveProperty('slug', first!.slug)
    expect(body).toHaveProperty('listeners')
    expect(body).toHaveProperty('opens')
    // `insights` is NOT user-derived, so it is never withheld — if it starts coming back null the
    // withholding logic has been applied too broadly.
    expect(typeof body.insights, '`insights` is corpus data and must never be withheld').toBe(
      'number',
    )
  })
})
