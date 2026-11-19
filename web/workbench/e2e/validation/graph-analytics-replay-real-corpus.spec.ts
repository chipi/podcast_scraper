/**
 * Tier-3 — graph analytics logging + replay against a real corpus.
 *
 * This is the acceptance scenario for the owned graph-analytics pipeline
 * (what users do → logged → replayable) tied to the detail-card navigation
 * actions of the embedded node view:
 *
 *   1. Open a graph, then drive the detail-card rail navigation via the DEV
 *      subject hook (``focusTopic`` / ``focusPerson``). Each focus change emits
 *      a ``graph_rail_nav`` analytics event (NodeDetail's ``props.nodeId``
 *      watcher), exactly as a user clicking related-topic / connections chips
 *      would.
 *   2. Flush the emitter and assert the batch was POSTed (204) AND persisted to
 *      ``<data_dir>/users/anon/graph_events.jsonl`` — "is all of it logged?".
 *   3. Feed the *real logged events* back into the replay engine
 *      (``__GIKG_REPLAY__``) and assert it reconstructs the trail + focus
 *      deterministically, matching a reference reconstruction that mirrors
 *      ``graphReplay.apply`` — "can we replay those logs well?".
 *
 * Runs auth-off (``make serve-for-validation`` sets ``APP_OAUTH_PROVIDER=none``),
 * so the persisted events land under the shared ``anon`` user bucket.
 *
 * Run: ``make ci-ui-validation CORPUS=/abs/path/to/your/corpus`` (needs a
 * running ``make serve-for-validation`` stack).
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import { expect, test, type Page, type TestInfo } from '@playwright/test'

import {
  SHELL_HEADING_RE,
  mainViewsNav,
  signInIsolated,
  statusBarCorpusPathInput,
} from '../helpers'

const CORPUS_PATH = process.env.CORPUS_PATH ?? ''

if (!CORPUS_PATH) {
  test.describe('Tier-3 graph analytics/replay (skipped — no CORPUS_PATH)', () => {
    test.skip(
      true,
      'Requires CORPUS_PATH. Run via `make ci-ui-validation CORPUS=/abs/path/to/your/corpus`.',
    )
  })
}

/* Auth was added to the app after this spec was written. Analytics events
 * used to land under ``<repo>/.app/users/anon/graph_events.jsonl`` (anon
 * bucket); now ``signInIsolated`` authenticates each spec so events land
 * under ``<CORPUS>/.app/users/u_<hash>/graph_events.jsonl`` — one file per
 * signed-in identity. We don't know the exact u_<hash> ahead of time, so
 * we read EVERY graph_events.jsonl under the users dir and filter by
 * session_id. The session_id from ``__GIKG_ANALYTICS__`` is unique per
 * test run, so filtering isolates our events even when a shared corpus
 * accumulates events from many runs. */
const APP_USERS_DIR = join(CORPUS_PATH, '.app', 'users')

function readAllUserEvents(): string {
  if (!existsSync(APP_USERS_DIR)) return ''
  const chunks: string[] = []
  for (const uidDir of readdirSync(APP_USERS_DIR)) {
    const p = join(APP_USERS_DIR, uidDir, 'graph_events.jsonl')
    if (existsSync(p)) chunks.push(readFileSync(p, 'utf-8'))
  }
  return chunks.join('\n')
}

interface LoggedEvent {
  action: string
  session_id?: string
  ts?: number
  to_id?: string
  target_id?: string
  id?: string
  [k: string]: unknown
}

/** Pull a couple of Topic + Person node ids from the corpus GI to drive with. */
function pickNavIds(): { topics: string[]; persons: string[] } {
  const gi: string[] = []
  const walk = (dir: string): void => {
    if (!existsSync(dir)) return
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const next = join(dir, e.name)
      if (e.isFile() && e.name.endsWith('.gi.json')) gi.push(next)
      else if (e.isDirectory()) walk(next)
    }
  }
  walk(CORPUS_PATH)
  const topics = new Set<string>()
  const persons = new Set<string>()
  for (const f of gi) {
    const body = JSON.parse(readFileSync(f, 'utf-8')) as {
      nodes?: Array<{ id?: string; type?: string }>
    }
    for (const n of body.nodes ?? []) {
      if (typeof n.id !== 'string') continue
      if (n.type === 'Topic' && n.id.includes('topic:')) topics.add(n.id)
      if (n.type === 'Person' && n.id.includes('person:')) persons.add(n.id)
    }
    if (topics.size >= 2 && persons.size >= 1) break
  }
  return { topics: [...topics].slice(0, 2), persons: [...persons].slice(0, 1) }
}

/** Reference reconstruction — mirrors ``graphReplay.apply`` + ``setTrail`` so the
 *  spec can assert the engine reproduces exactly what the logged events imply. */
function reconstruct(events: LoggedEvent[]): { trail: string[]; focus: string | null } {
  const raw: string[] = []
  let focus: string | null = null
  for (const e of events) {
    if (e.action === 'graph_recenter' && typeof e.target_id === 'string') {
      raw.length = 0
      focus = e.target_id
    } else if (e.action === 'graph_rail_nav' && typeof e.to_id === 'string') {
      raw.push(e.to_id)
      focus = e.to_id
    } else if (e.action === 'graph_node_tap' && typeof e.id === 'string') {
      focus = e.id
    }
  }
  // setTrail: trim + dedup + cap to TRAIL_BUDGET (28).
  const trail = [...new Set(raw.map((s) => s.trim()).filter(Boolean))].slice(-28)
  return { trail, focus }
}

async function openGraphFromLibrary(page: Page, testInfo: TestInfo): Promise<void> {
  await signInIsolated(page, 'graph-analytics-replay', testInfo)
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 30_000 })
  const input = statusBarCorpusPathInput(page)
  await input.waitFor({ state: 'visible', timeout: 15_000 })
  await input.fill(CORPUS_PATH)
  await input.press('Enter').catch(() => {})
  await page.waitForTimeout(1_500)
  await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
  const firstRow = page
    .getByRole('button', { name: /, / })
    .filter({ hasNotText: 'Open in graph' })
    .first()
  await firstRow.click({ timeout: 15_000 })
  await page.getByRole('button', { name: 'Open in graph' }).first().click()
  // Wait for the FSM to settle so the handoff recenter has been emitted.
  await page
    .waitForFunction(
      () => {
        const w = window as unknown as { __GIKG_FSM__?: { state: string } }
        return w.__GIKG_FSM__?.state === 'ready'
      },
      undefined,
      { timeout: 30_000 },
    )
    .catch(() => void 0)
}

test.describe('Tier-3 — graph analytics logging + replay (real corpus)', () => {
  test.setTimeout(120_000)

  test('detail-card nav is logged to graph_events.jsonl and replays faithfully', async ({
    page,
  }, testInfo) => {
    const { topics, persons } = pickNavIds()
    test.skip(
      topics.length < 1 || persons.length < 1,
      'corpus lacks Topic + Person GI nodes to drive the nav session',
    )
    // The detail-card nav sequence we drive (each focus → one graph_rail_nav).
    const navSeq: Array<{ kind: 'topic' | 'person'; id: string }> = [
      { kind: 'topic', id: topics[0] },
      { kind: 'person', id: persons[0] },
      ...(topics[1] ? [{ kind: 'topic' as const, id: topics[1] }] : []),
    ]

    // Capture the fire-and-forget analytics POSTs.
    const postStatuses: number[] = []
    page.on('response', (r) => {
      const req = r.request()
      if (req.method() === 'POST' && r.url().includes('/api/app/graph-events')) {
        postStatuses.push(r.status())
      }
    })

    await openGraphFromLibrary(page, testInfo)

    // The emitter's session id — used to isolate THIS run's events from the
    // append-only log's prior-session rows.
    const sessionId = await page.evaluate(() => {
      const w = window as unknown as { __GIKG_ANALYTICS__?: { sessionId: string } }
      if (!w.__GIKG_ANALYTICS__) throw new Error('__GIKG_ANALYTICS__ hook missing (DEV/e2e only)')
      return w.__GIKG_ANALYTICS__.sessionId
    })

    // Drive the detail-card navigation via the subject hook: each focus changes
    // NodeDetail's nodeId → emits graph_rail_nav (same path as clicking a
    // related-topic / connections chip in the rail).
    for (const step of navSeq) {
      await page.evaluate(
        ({ kind, id }) => {
          const w = window as unknown as {
            __GIKG_SUBJECT__?: { focusTopic: (i: string) => void; focusPerson: (i: string) => void }
          }
          const hook = w.__GIKG_SUBJECT__
          if (!hook) throw new Error('__GIKG_SUBJECT__ hook missing')
          if (kind === 'topic') hook.focusTopic(id)
          else hook.focusPerson(id)
        },
        step,
      )
      await page.waitForTimeout(300)
    }

    // Flush the emitter now (steady state uses a 10s timer / tab-hide).
    await page.evaluate(() => {
      ;(window as unknown as { __GIKG_ANALYTICS__: { flush: () => void } }).__GIKG_ANALYTICS__.flush()
    })

    // (A) Logged over the wire: at least one batch POSTed, every one 204.
    await expect.poll(() => postStatuses.length, { timeout: 10_000 }).toBeGreaterThan(0)
    expect(postStatuses.every((s) => s === 204)).toBe(true)

    // (B) Persisted server-side: read the per-user jsonl(s), isolate this
    // session. See ``readAllUserEvents`` — with auth on, events land under
    // ``<CORPUS>/.app/users/u_<hash>/graph_events.jsonl`` (one file per
    // signed-in identity), so we concatenate every file and filter by
    // session_id.
    await expect
      .poll(() => readAllUserEvents(), { timeout: 10_000 })
      .toContain(sessionId)
    const allEvents = readAllUserEvents()
      .trim()
      .split('\n')
      .filter(Boolean)
      .map((l) => JSON.parse(l) as LoggedEvent)
    const sessionEvents = allEvents
      .filter((e) => e.session_id === sessionId)
      .sort((a, b) => (a.ts ?? 0) - (b.ts ?? 0))

    expect(sessionEvents.length).toBeGreaterThan(0)
    // Every driven detail-card nav target is present in the log. The rail records
    // the graph-layer-prefixed id (``g:topic:…``) once the node loads into the
    // slice, so compare on the bare corpus id (strip the ``g:`` / ``k:`` layer).
    const stripLayer = (id: string): string => id.replace(/^(g|k):/, '')
    const railNavTargets = new Set(
      sessionEvents
        .filter((e) => e.action === 'graph_rail_nav' && typeof e.to_id === 'string')
        .map((e) => stripLayer(e.to_id as string)),
    )
    for (const step of navSeq) {
      expect(
        railNavTargets.has(step.id),
        `graph_rail_nav for ${step.id} not logged (got ${[...railNavTargets].join(', ')})`,
      ).toBe(true)
    }
    // Single session id across the whole batch (per-session isolation holds).
    expect(new Set(sessionEvents.map((e) => e.session_id)).size).toBe(1)

    // eslint-disable-next-line no-console
    console.log(
      '[Tier-3 analytics] logged',
      sessionEvents.length,
      'events for session',
      sessionId,
      '— actions:',
      JSON.stringify(
        sessionEvents.reduce<Record<string, number>>((acc, e) => {
          acc[e.action] = (acc[e.action] ?? 0) + 1
          return acc
        }, {}),
      ),
    )

    // (C) Replay the REAL logged events and assert deterministic reconstruction.
    const expectedEnd = reconstruct(sessionEvents)
    await page.evaluate(
      ({ sid, evs }) => {
        ;(
          window as unknown as {
            __GIKG_REPLAY__: { load: (id: string, e: unknown[]) => void }
          }
        ).__GIKG_REPLAY__.load(sid, evs)
      },
      { sid: sessionId, evs: sessionEvents },
    )

    // Step to the end → trail + focus match the reference reconstruction.
    const atEnd = await page.evaluate((n) => {
      const r = (
        window as unknown as {
          __GIKG_REPLAY__: {
            setStep: (n: number) => void
            active: boolean
            trailIds: string[]
            focus: string | null
          }
        }
      ).__GIKG_REPLAY__
      r.setStep(n)
      return { active: r.active, trail: r.trailIds, focus: r.focus }
    }, sessionEvents.length)
    expect(atEnd.active).toBe(true)
    expect(atEnd.trail).toEqual(expectedEnd.trail)
    expect(atEnd.focus).toBe(expectedEnd.focus)

    // Scrub back to a mid step → the trail is the reconstruction of the prefix
    // (deterministic rewind, not just forward accumulation).
    const mid = Math.max(1, Math.floor(sessionEvents.length / 2))
    const expectedMid = reconstruct(sessionEvents.slice(0, mid))
    const atMid = await page.evaluate((n) => {
      const r = (
        window as unknown as {
          __GIKG_REPLAY__: { setStep: (n: number) => void; trailIds: string[]; focus: string | null }
        }
      ).__GIKG_REPLAY__
      r.setStep(n)
      return { trail: r.trailIds, focus: r.focus }
    }, mid)
    expect(atMid.trail).toEqual(expectedMid.trail)
    expect(atMid.focus).toBe(expectedMid.focus)

    // Exit clears the replay session + trail.
    const afterExit = await page.evaluate(() => {
      const r = (
        window as unknown as {
          __GIKG_REPLAY__: { exit: () => void; active: boolean; trailIds: string[] }
        }
      ).__GIKG_REPLAY__
      r.exit()
      return { active: r.active, trail: r.trailIds }
    })
    expect(afterExit.active).toBe(false)
    expect(afterExit.trail).toEqual([])
  })
})
