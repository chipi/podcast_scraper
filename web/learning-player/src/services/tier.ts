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

// The developer's local dev API, reached over the TAILNET (not localhost) so it works from a physical
// device on any network — a roaming phone has no `localhost` route to the laptop. Fronted by
// `tailscale serve` on this host, which gives a MagicDNS HTTPS cert (tailnet-only) so iOS ATS accepts
// it with no cleartext exception. Requires `tailscale serve` to proxy this host's :443 → the dev API
// (currently → 127.0.0.1:8080; point it at wherever `make serve-app` / the api serves /api/app).
//
// Three rungs, in this order:
//   1. VITE_DEV_API_BASE   explicit, from the gitignored `.env.mobile` (see .env.mobile.example).
//   2. __DEV_API_BASE__    derived at build time from the BUILD HOST's own tailnet DNS name
//                          (vite.config.ts § resolveDevApiBase).
//   3. loopback            the simulator, which shares the host's loopback.
//
// The tailnet host is never hardcoded: it is a personal machine name, it trips the operator
// identifier deny-list, and it bakes a private hostname into every shipped bundle.
//
// Rung 2 is what makes a PHYSICAL device work with no configuration. A device has no route to the
// host's loopback — which is the whole reason a tailnet address was wanted in the first place —
// and there are two dev machines the checkout moves between, so naming one of them by hand is
// both a setup step and wrong half the time. Deriving from the machine doing the build is right
// by construction: that is the host `tailscale serve` fronts.
const DEV_API_BASE =
  import.meta.env.VITE_DEV_API_BASE ||
  (typeof __DEV_API_BASE__ === 'string' ? __DEV_API_BASE__ : '') ||
  'http://127.0.0.1:8080/api/app'
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

/**
 * Absolutise a media URL the API handed us (artwork, images).
 *
 * The API returns these RELATIVE (`/api/app/artwork?ref=…`). On web that is correct — the app and
 * the API share an origin. On native the document origin is `capacitor://localhost`, so the same
 * string resolves into the app bundle and every image 404s. `fetch` was never affected because
 * `api.ts` prefixes an absolute BASE itself; `<img src>` had nothing doing that for it.
 *
 * Absolute URLs (including `data:`, `file:` and `capacitor:`) are returned untouched, so a locally
 * downloaded artwork path passes straight through.
 */
export function resolveMediaUrl(url: string | null | undefined): string | null {
  if (!url) return null
  if (/^[a-z][a-z0-9+.-]*:/i.test(url)) return url
  const base = resolveApiBase()
  // Web: origin-relative is already right, and BASE is itself relative.
  if (!/^https?:/i.test(base)) return url
  try {
    return new URL(url, base).toString()
  } catch {
    return url
  }
}

// Coming-soon gate credential (player.caddy §@authed fallback). The prod edge gates /api/app/* behind
// a secret cookie; the deliberate fallback is that a valid Basic-auth `Authorization` header also
// passes. We bake `VITE_PREVIEW_BASIC_AUTH` (= base64("user:pass")) from the gitignored .env.mobile so
// the native prod tier can reach the gated API while the site is pre-launch. It is attached ONLY on an
// internal build's prod tier — never on the laptop dev tier, and never in a prod-locked release
// (tierSwitchEnabled() is false there, so the secret is not baked into a shipped app).
const PREVIEW_BASIC_AUTH = import.meta.env.VITE_PREVIEW_BASIC_AUTH

/**
 * `Authorization` value that clears the prod coming-soon gate, or null when it doesn't apply
 * (web, release, or the dev/laptop tier). api.ts attaches it only when no user Bearer token is set —
 * the gate's Basic-auth and a user session both use `Authorization`, so a signed-in session takes
 * precedence (open reads work gate-only; per-user writes need the not-yet-wired native login).
 */
export function resolveGateAuthHeader(): string | null {
  if (!tierSwitchEnabled()) return null
  if (getTier() !== 'prod') return null
  return PREVIEW_BASIC_AUTH ? `Basic ${PREVIEW_BASIC_AUTH}` : null
}

// The gate's `cl_preview` cookie value (from the /preview handshake), baked from .env.mobile. Set into
// the native cookie jar on prod tier so caddy's @preview_ok opens the gate on the COOKIE — which it
// checks BEFORE @authed — leaving the `Authorization` header free for the signed-in user's Bearer.
// This is what lets sign-in work: Basic-in-Authorization collides with Bearer; the cookie doesn't.
const PREVIEW_COOKIE = import.meta.env.VITE_PREVIEW_COOKIE

/** `cl_preview` cookie value for the native prod tier (internal builds only), or null. */
export function resolveGateCookie(): string | null {
  if (!tierSwitchEnabled()) return null
  if (getTier() !== 'prod') return null
  return PREVIEW_COOKIE || null
}
