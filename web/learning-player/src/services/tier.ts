/**
 * Runtime dev↔prod target switch for the native shell (#1310, mobile-app-guide §5 / Orrery ADR-083).
 *
 * The podcast estate has **no staging** (ADR-126) — the switch is dev↔prod only:
 *   - **prod**: the live player API + the prod GlitchTip/Umami baked at build time.
 *   - **dev**:  the developer's LOCAL full setup — `make serve-app` (vite :5174, proxying /api →
 *     :8000) — plus the tailnet dev GlitchTip/Umami. "Load everything from the local machine."
 *
 * Web builds ignore this entirely (auth rides the same-origin cookie; `resolveApiBase()` returns the
 * origin-relative base). The switch only applies inside the Capacitor shell, and only when the build
 * is an INTERNAL one (`__MOBILE_INTERNAL__`) — release builds are prod-locked (the toggle is tree-
 * shaken out), so a shipped app can never point at a developer's laptop.
 *
 * Only the API base + telemetry channels (GlitchTip error DSN) follow the switch. **Umami stays the
 * prod site** so UX analytics stay unified across tiers (operator's call). The WebView shell assets
 * are always the bundled `dist` (a pure SPA — no prerendered origin to repoint; guide §5).
 */
import { Capacitor } from '@capacitor/core'

export type Tier = 'dev' | 'prod'

const TIER_KEY = 'lp_tier'

// Build flag: true for internal / simulator / TestFlight builds (the switch is available); false in
// release (prod-locked, toggle tree-shaken). Defaults to DEV when the define is absent.
declare const __MOBILE_INTERNAL__: boolean | undefined
export function isInternalBuild(): boolean {
  return typeof __MOBILE_INTERNAL__ !== 'undefined' ? !!__MOBILE_INTERNAL__ : import.meta.env.DEV
}

/** Whether the dev/prod switch applies at all (native shell + an internal build). */
export function tierSwitchEnabled(): boolean {
  return Capacitor.isNativePlatform() && isInternalBuild()
}

/** Current tier — persisted; always 'prod' when the switch is disabled (web / release). */
export function getTier(): Tier {
  if (!tierSwitchEnabled()) return 'prod'
  try {
    return localStorage.getItem(TIER_KEY) === 'dev' ? 'dev' : 'prod'
  } catch {
    return 'prod'
  }
}

/** Persist the selected tier (caller reloads so api.ts + telemetry re-resolve). No-op if disabled. */
export function setTier(tier: Tier): void {
  if (!tierSwitchEnabled()) return
  try {
    localStorage.setItem(TIER_KEY, tier)
  } catch {
    /* storage disabled — ignore */
  }
}

// The developer's local full setup (make serve-app): vite on :5174 proxies /api → :8000.
const DEV_API_BASE = 'http://localhost:5174/api/app'
// Live player API (public consumer plane, same-origin on web). Overridable via VITE_API_BASE_URL.
const PROD_API_BASE = 'https://closelistening.app/api/app'

/**
 * The API base for the current context:
 *   - web: origin-relative (`/api/app`) unless a build baked VITE_API_BASE_URL.
 *   - native prod (or release): VITE_API_BASE_URL if baked, else the live player API.
 *   - native dev: the local machine.
 */
export function resolveApiBase(): string {
  const baked = import.meta.env.VITE_API_BASE_URL
  if (!Capacitor.isNativePlatform()) return baked || '/api/app'
  if (getTier() === 'dev') return DEV_API_BASE
  return baked || PROD_API_BASE
}
