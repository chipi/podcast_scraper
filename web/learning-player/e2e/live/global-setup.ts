/**
 * Live-smoke readiness + warmup — wait for the deployed app to be UP, then pay its cold start once.
 *
 * ## Two different jobs, and this used to do only the second
 *
 * **Readiness** is "is the surface answering yet". **Warmup** is "the first hit is expensive, so
 * absorb it before the assertions start". They are not the same, and doing only the warmup left a
 * real gap: the deploy restarts containers and the smoke starts immediately after, so if the app is
 * still coming up the warmup requests fail, get swallowed (by design — see below), and the suite
 * begins against a surface that is not there yet. Every early test then fails for a reason that has
 * nothing to do with what it asserts.
 *
 * So this now POLLS `/api/health` until it answers, with a deadline, before warming anything.
 *
 * 2026-08-27 (the warmup half): the smoke ran ~3 min after a restart and the episode page missed
 * its 15s budget three times in a row — retries all landed in the same cold window, a false red on
 * a healthy deploy. The heavy first-hit costs are server-side (catalog scan + slug index + search
 * model load), so one round of warmup absorbs them for the whole suite.
 *
 * ## Why failures here are still swallowed
 *
 * A down app should be reported by the TESTS, with their own assertions and traces — not by an
 * opaque setup crash. If readiness never arrives this logs loudly and returns; the specs then fail
 * with real messages against real URLs, which is the diagnosis you actually want.
 *
 * ## Auth: two doors, since RFC-120
 *
 * The warmup used to send only the gate's Basic. That stopped reaching anything the moment #1940
 * put `Depends(get_current_user)` on the content API — `/api/app/episodes` returned 401, the
 * episode lookup below yielded nothing, and the warmup quietly degraded to a couple of endpoints
 * while still logging success. It mints the app session too now, exactly like the specs.
 */
import type { FullConfig } from '@playwright/test'
import { bearer, canMintSession, gatePass, gateUser } from './session'

/** How long to wait for the surface to answer before giving up and letting the tests report it. */
const READY_DEADLINE_MS = 120_000
const READY_INTERVAL_MS = 3_000

export default async function globalSetup(_config: FullConfig): Promise<void> {
  const baseURL = process.env.LIVE_BASE_URL || 'https://closelistening.app'
  if (!gatePass) return // gated specs will skip anyway; nothing to wait for or warm

  const basic = 'Basic ' + Buffer.from(`${gateUser}:${gatePass}`).toString('base64')

  // The gate's PRIMARY mechanism is the `cl_preview` COOKIE, not the Basic challenge: sending only
  // Basic to /api/health returns the coming-soon HTML, which is what the first version of this poll
  // did — it spent its whole 120s deadline failing to parse `<!doctype html>` as JSON and then gave
  // up on every run. `fetch` keeps no cookie jar, so the handshake is explicit: GET /preview with
  // Basic, keep what it Set-Cookies, and send that from then on.
  let gateCookie = ''
  try {
    const gate = await fetch(`${baseURL}/preview`, {
      headers: { Authorization: basic },
      redirect: 'manual',
      signal: AbortSignal.timeout(30_000),
    })
    gateCookie = (gate.headers.getSetCookie?.() ?? [])
      .map((c) => c.split(';')[0])
      .join('; ')
    if (!gateCookie) console.log('[live-smoke readiness] /preview set no cookie — falling back to Basic')
  } catch (err) {
    console.log(`[live-smoke readiness] /preview handshake failed: ${String(err)}`)
  }

  const gateHeaders: Record<string, string> = gateCookie
    ? { Cookie: gateCookie }
    : { Authorization: basic }
  // An explicit Bearer overrides the Basic that clears the gate, so app-authenticated warmups can
  // only be sent once the gate is satisfied some other way. `/api/health` sits behind the gate but
  // needs no session, so readiness uses Basic and the content warmups use the session.
  const appHeaders: Record<string, string> = canMintSession
    ? { ...gateHeaders, ...bearer() }
    : gateHeaders

  // --- readiness -------------------------------------------------------------------------------
  const started = Date.now()
  let ready = false
  while (Date.now() - started < READY_DEADLINE_MS) {
    try {
      const resp = await fetch(`${baseURL}/api/health`, {
        headers: gateHeaders,
        signal: AbortSignal.timeout(10_000),
      })
      if (resp.ok) {
        const text = await resp.text()
        if (text.trimStart().startsWith('<')) {
          // HTML from /api/health means the gate answered instead of the app — a credentials
          // problem here, never a health problem. Naming it stops the next person reading this as
          // an outage.
          console.log('[live-smoke readiness] gate returned HTML for /api/health — creds not accepted')
          await new Promise((r) => setTimeout(r, READY_INTERVAL_MS))
          continue
        }
        const body = JSON.parse(text) as { status?: string }
        if (body.status === 'ok') {
          ready = true
          break
        }
        console.log(`[live-smoke readiness] health says "${body.status}" — still waiting`)
      } else {
        console.log(`[live-smoke readiness] health HTTP ${resp.status} — still waiting`)
      }
    } catch (err) {
      console.log(`[live-smoke readiness] health unreachable (${String(err)}) — still waiting`)
    }
    await new Promise((r) => setTimeout(r, READY_INTERVAL_MS))
  }

  const waited = Date.now() - started
  if (!ready) {
    // Loud, and NOT fatal. The specs will now fail with their own assertions, which say far more
    // than "global setup threw" ever could.
    console.log(
      `[live-smoke readiness] NOT ready after ${waited}ms — running anyway so the specs report ` +
        `the real failure with traces`,
    )
    return
  }
  console.log(`[live-smoke readiness] surface ready after ${waited}ms`)

  // --- warmup ----------------------------------------------------------------------------------
  const warmStarted = Date.now()
  try {
    const episodes = await fetch(`${baseURL}/api/app/episodes?page_size=15`, {
      headers: appHeaders,
      signal: AbortSignal.timeout(60_000),
    })
    if (!episodes.ok) {
      // Worth saying out loud rather than degrading in silence — this is exactly how the auth
      // change went unnoticed here.
      console.log(`[live-smoke warmup] episode list HTTP ${episodes.status} — warming less`)
    }
    const list = episodes.ok
      ? ((await episodes.json()) as {
          items?: Array<{ slug: string; status: string; has_bridge: boolean }>
        })
      : {}
    const ep = list.items?.find((e) => e.status === 'ready' && e.has_bridge)
    const warmups = [
      `${baseURL}/api/app/podcasts`,
      `${baseURL}/api/app/theme-clusters?limit=3`,
      ...(ep ? [`${baseURL}/api/app/episodes/${ep.slug}`] : []),
    ]
    await Promise.allSettled(
      warmups.map((u) =>
        fetch(u, { headers: appHeaders, signal: AbortSignal.timeout(60_000) }),
      ),
    )
    console.log(
      `[live-smoke warmup] done in ${Date.now() - warmStarted}ms (${warmups.length + 1} requests)`,
    )
  } catch (err) {
    console.log(`[live-smoke warmup] non-fatal: ${String(err)} (${Date.now() - warmStarted}ms)`)
  }
}
