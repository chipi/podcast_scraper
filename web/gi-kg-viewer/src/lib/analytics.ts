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

/** NO LITERAL TARGET (operator, 2026-10-03) — the same fix the player's analytics and every
 *  Sentry DSN got. This used to fall back, in `vite dev`, to a hardcoded `operator-dev` site at the
 *  tailnet host `homelab:3001`: a personal hostname baked into the source, which does not resolve
 *  from every machine and is refused on the homelab box itself (services there answer on loopback).
 *
 *  Both rungs now come from the environment only:
 *    `VITE_UMAMI_SRC` / `VITE_UMAMI_WEBSITE_ID`         — prod, docker build-args
 *    `VITE_UMAMI_SRC_DEV` / `VITE_UMAMI_WEBSITE_ID_DEV` — the dev rung, from a local .env
 *  `VITE_UMAMI_SRC*` is the FULL tracking-script URL (same convention as the player), injected
 *  verbatim. With none set, analytics is a true no-op. */
const devRungEnabled = import.meta.env.DEV && import.meta.env.VITE_ANALYTICS_OFF !== '1'

function umamiSrc(): string {
  return (
    (import.meta.env.VITE_UMAMI_SRC as string) ||
    (devRungEnabled ? (import.meta.env.VITE_UMAMI_SRC_DEV as string) : '') ||
    ''
  )
}
function umamiWebsiteId(): string {
  return (
    (import.meta.env.VITE_UMAMI_WEBSITE_ID as string) ||
    (devRungEnabled ? (import.meta.env.VITE_UMAMI_WEBSITE_ID_DEV as string) : '') ||
    ''
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
 *  History API, so SPA route changes auto-track without extra wiring. */
export function initAnalytics(): void {
  if (typeof document === 'undefined') return
  const src = umamiSrc()
  const websiteId = umamiWebsiteId()
  if (!src || !websiteId) return // fork-silent (non-dev build, no env) by construction
  if (document.querySelector('script[data-umami-installed]')) return
  const s = document.createElement('script')
  s.defer = true
  s.src = src
  s.setAttribute('data-website-id', websiteId)
  s.setAttribute('data-umami-installed', '1')
  document.head.appendChild(s)
}

type UmamiGlobal = {
  track?: (name: string, props?: Record<string, unknown>) => void
}

/** Track a custom event. Name is constrained to the registry (typos are compile
 *  errors). Safe before the Umami script loads (it queues), and a no-op when
 *  analytics is disabled. Drop-in for the old `posthog.capture(name, props)`. */
export function track(name: EventName, props?: Record<string, unknown>): void {
  if (!analyticsEnabled()) return
  if (typeof window === 'undefined') return
  const u = (window as unknown as { umami?: UmamiGlobal }).umami
  u?.track?.(name, props)
}
