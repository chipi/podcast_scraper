/**
 * Umami analytics for the operator viewer — cookieless, PII-free custom-event
 * tracking. Replaces PostHog Cloud (removed 2026-07-24): the whole estate now
 * runs on the self-hosted Umami (the player + orrery already do), so dev/prod
 * telemetry is separable and nothing leaves for a third-party cloud.
 *
 * ── Enablement (mirrors the player, web/learning-player/src/main.ts) ──────────
 * Gated on VITE_UMAMI_SRC (the tracking-script URL) + VITE_UMAMI_WEBSITE_ID
 * being baked at build time. A prod build sets both (docker build-args) → live.
 * A non-dev build with neither → true no-op (fork-silent by construction).
 *
 * ── Dev rung of the env ladder (dev → prod; the operator has no staging) ──────
 * In `vite dev`, with no override, events go to the dedicated `operator-dev`
 * Umami site via the Tailscale host `homelab` — NO fixed IP. Only a device ON
 * the tailnet (the operator's machine) resolves `homelab`, so a stranger who
 * checks out + runs the repo sends nothing (the request just fails). The
 * website id is a public browser id (ships in the bundle) — safe to commit.
 * `VITE_ANALYTICS_OFF=1` hard-disables the dev default (set by the vitest +
 * playwright configs so e2e / unit runs never emit).
 *
 * ── Event registry (single source of truth) ──────────────────────────────────
 * Every custom event name lives in `EVENT_NAMES`; `track()` only accepts those,
 * so a typo or an ad-hoc name is a compile error. Names are the verbatim ones
 * the viewer emitted under PostHog — re-homed onto Umami, dashboards keep working.
 */

/** The `operator-dev` Umami site, reached over the tailnet in `vite dev`.
 *  `VITE_UMAMI_SRC` is the FULL tracking-script URL (same convention as the
 *  player) — injected verbatim, never suffixed. */
const DEV_UMAMI_SRC = 'http://homelab:3001/script.js'
const DEV_UMAMI_WEBSITE_ID = '3bc8920d-d2bb-4a7f-a16b-057b1ed944c4'

/** Dev default is live in `vite dev` unless a test runner explicitly opted out. */
const devDefaultEnabled = import.meta.env.DEV && import.meta.env.VITE_ANALYTICS_OFF !== '1'

function umamiSrc(): string {
  return (import.meta.env.VITE_UMAMI_SRC as string) || (devDefaultEnabled ? DEV_UMAMI_SRC : '')
}
function umamiWebsiteId(): string {
  return (
    (import.meta.env.VITE_UMAMI_WEBSITE_ID as string) ||
    (devDefaultEnabled ? DEV_UMAMI_WEBSITE_ID : '')
  )
}

/** The canonical viewer event vocabulary. `track()` accepts only these. */
export const EVENT_NAMES = [
  'main_tab_switched',
  'graph_corpus_synced',
  'episode_focused',
  'search_run',
  'graph_handoff_stuck',
  'graph_handoff_started',
  'graph_handoff_failed',
  'graph_handoff_applied',
  'left_panel_surface_switched',
  'corpus_path_changed',
] as const

export type EventName = (typeof EVENT_NAMES)[number]

/** Analytics fires when a script URL + website id resolve for the current rung —
 *  a build-time env override (prod) or the `vite dev` default. Fork-silent by
 *  construction: a non-dev build with no env resolves neither, so no script is
 *  injected and every `track()` is a no-op. */
function analyticsEnabled(): boolean {
  return !!umamiSrc() && !!umamiWebsiteId()
}

/** Inject the self-hosted Umami `<script>` exactly once, only when enabled.
 *  Idempotent. Call from `main.ts` after the app is created. Umami hooks the
 *  History API, so SPA route changes auto-track without extra wiring.
 *
 *  Injection is deferred until AFTER the window `load` event. A `<script>` in
 *  the head — even `defer` — keeps `load` pending until its fetch settles, so
 *  an unreachable analytics host holds the whole page "loading" for the full
 *  network timeout. Observed on the 2026-09-07 DR drill: the Playwright browser
 *  runs on a GitHub runner with no tailnet route to the homelab umami host, the
 *  request never settled (`status=-1`), and `page.goto` blocked 92.5s of a 120s
 *  budget while every other resource finished in under 2s. Any client that
 *  cannot reach the analytics host pays the same stall, so this is fixed here
 *  rather than worked around in the test. Analytics must never gate readiness. */
export function initAnalytics(): void {
  if (typeof document === 'undefined') return
  const src = umamiSrc()
  const websiteId = umamiWebsiteId()
  if (!src || !websiteId) return // fork-silent (non-dev build, no env) by construction

  const inject = (): void => {
    // Re-check inside the callback: two initAnalytics() calls before `load`
    // would otherwise queue two listeners and inject twice.
    if (document.querySelector('script[data-umami-installed]')) return
    const s = document.createElement('script')
    s.defer = true
    s.src = src
    s.setAttribute('data-website-id', websiteId)
    s.setAttribute('data-umami-installed', '1')
    document.head.appendChild(s)
  }

  if (document.readyState === 'complete') {
    inject()
  } else {
    window.addEventListener('load', inject, { once: true })
  }
}

type UmamiGlobal = {
  track?: (name: string, props?: Record<string, unknown>) => void
}

/** Track a custom event. Name is constrained to the registry (typos are compile
 *  errors). Safe to call at any time, but NOT queued: until the Umami script has
 *  executed, `window.umami` is undefined and the event is dropped. Since the
 *  script is now injected after `load`, events fired during start-up are lost —
 *  an accepted trade for never stalling page readiness. Pageviews are unaffected
 *  (Umami records one on init). Also a no-op when analytics is disabled. */
export function track(name: EventName, props?: Record<string, unknown>): void {
  if (!analyticsEnabled()) return
  if (typeof window === 'undefined') return
  const u = (window as unknown as { umami?: UmamiGlobal }).umami
  u?.track?.(name, props)
}
