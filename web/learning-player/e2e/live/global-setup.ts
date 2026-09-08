/**
 * Live-smoke readiness + warmup — wait for the deployed app to be UP, then pay its cold start once.
 *
 * ## Readiness targets the teaser endpoint, and needs no credentials
 *
 * `/api/app/discover` is the entire anonymous surface: Caddy's `@teaser` block passes it with no
 * cookie, Basic or Bearer (`infra/caddy/player.caddy`), and the backend serves it through
 * `get_optional_user`, clamping anonymous callers rather than rejecting them. It also answers
 * **503 until the corpus root is loaded** (`corpus_root_or_503`), which is exactly the "is this
 * surface actually ready" signal — a health check that only proves the process is alive is weaker.
 *
 * An earlier version polled `/api/health` and NEVER succeeded, burning its whole deadline on every
 * run. That route is not reachable from outside at all: the learning-app container's nginx proxies
 * only `/api/app/`, `/api/app/auth/` and the OAuth well-known, so `/api/health` falls into the SPA
 * catch-all and returns `index.html` with a 200. The HTML that came back was the app shell, not the
 * gate — the credentials were fine, the route simply does not exist out there. The log line then
 * blamed the credentials, which was wrong and would have misled whoever read it next.
 *
 * Hence the content-type check below: an HTML body means something served the SPA instead of the
 * API, and that is never a ready signal whatever status it carries.
 *
 * ## Warmup is separate, and must never be skipped for readiness' sake
 *
 * 2026-08-27: the smoke ran ~3 min after a restart and the episode page missed its 15s budget three
 * times running — retries all landed in the same cold window, a false red on a healthy deploy. The
 * heavy first-hit costs are server-side (catalog scan, slug index, search model load), so one round
 * of warmup absorbs them for the whole suite.
 *
 * A previous revision `return`ed when readiness was not confirmed, silently disabling that warmup on
 * every run. Readiness is an OPTIMISATION over the warmup; failing to establish it must never remove
 * something that already worked.
 *
 * ## Nothing here is fatal
 *
 * A down app should be reported by the TESTS, with their own assertions, traces and screenshots —
 * not by an opaque setup crash.
 */
import type { FullConfig } from '@playwright/test'
import { bearer, canMintSession } from './session'

/** Short on purpose: a poll that cannot connect should be an annoyance, not a tax on every deploy. */
const READY_DEADLINE_MS = 30_000
const READY_INTERVAL_MS = 3_000

export default async function globalSetup(_config: FullConfig): Promise<void> {
  const baseURL = process.env.LIVE_BASE_URL || 'https://closelistening.app'

  // --- readiness (no credentials needed) -------------------------------------------------------
  const started = Date.now()
  let ready = false
  while (Date.now() - started < READY_DEADLINE_MS) {
    try {
      const resp = await fetch(`${baseURL}/api/app/discover?limit=1`, {
        signal: AbortSignal.timeout(10_000),
      })
      const contentType = resp.headers.get('content-type') ?? ''
      if (resp.ok && contentType.includes('application/json')) {
        ready = true
        break
      }
      if (resp.ok) {
        // 200 + HTML means the SPA catch-all answered, not the API — a routing fact, never a
        // credentials one. Naming which keeps the next reader out of the hole this replaced.
        console.log(`[live-smoke readiness] non-JSON from discover (${contentType}) — still waiting`)
      } else {
        // A 503 here is the corpus still loading, which is precisely what this poll is for.
        console.log(`[live-smoke readiness] discover HTTP ${resp.status} — still waiting`)
      }
    } catch (err) {
      console.log(`[live-smoke readiness] discover unreachable (${String(err)}) — still waiting`)
    }
    await new Promise((r) => setTimeout(r, READY_INTERVAL_MS))
  }

  const waited = Date.now() - started
  console.log(
    ready
      ? `[live-smoke readiness] surface ready after ${waited}ms`
      : `[live-smoke readiness] not confirmed after ${waited}ms — warming and running anyway; ` +
          `the specs report real failures with traces`,
  )

  // --- warmup ----------------------------------------------------------------------------------
  //
  // The content endpoints are auth-required (login-first), and Caddy's `@bearer` block passes an
  // `Authorization: Bearer` on `/api/app/*` with no gate cookie — so the session alone is both
  // necessary and sufficient here. Without one there is nothing to warm.
  if (!canMintSession) {
    console.log('[live-smoke warmup] no session secret — skipping (the gated specs skip too)')
    return
  }

  const warmStarted = Date.now()
  const headers = bearer()
  try {
    const episodes = await fetch(`${baseURL}/api/app/episodes?page_size=15`, {
      headers,
      signal: AbortSignal.timeout(60_000),
    })
    if (!episodes.ok) {
      // Said out loud rather than degrading in silence — quiet degradation is how this warmup kept
      // "succeeding" after the auth change while warming almost nothing.
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
      warmups.map((u) => fetch(u, { headers, signal: AbortSignal.timeout(60_000) })),
    )
    console.log(
      `[live-smoke warmup] done in ${Date.now() - warmStarted}ms (${warmups.length + 1} requests)`,
    )
  } catch (err) {
    console.log(`[live-smoke warmup] non-fatal: ${String(err)} (${Date.now() - warmStarted}ms)`)
  }
}
