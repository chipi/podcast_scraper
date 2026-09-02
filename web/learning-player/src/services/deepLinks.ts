/**
 * Deep links: `closelistening://<target>/<id>` → an in-app route (#1925).
 *
 * The scheme was already registered for the OAuth callback, and `appUrlOpen` only ever looked for
 * a token — so every other link the app could plausibly receive was silently dropped. This is the
 * primitive the rest of the product needs to point AT something: a shared episode, a recap linking
 * the episodes it summarises, a notification about a show, an MCP tool answering with a citation.
 *
 * Two properties matter more than the parsing:
 *
 * 1. **The mapping is a closed allow-list.** An inbound URL is attacker-controllable — anything on
 *    the device can open one — so this never turns a URL path into a router path directly. It
 *    matches a known target, validates the id's SHAPE, and returns a route built from named
 *    constants. Nothing else routes anywhere.
 * 2. **It is pure.** No router, no store, no Capacitor import: it takes a string and returns a
 *    location or null. That is what makes it testable on the web tier, where the native listener
 *    it feeds cannot run at all.
 *
 * Route GUARDS still apply — a link to an auth-gated route lands on the sign-in gate exactly as an
 * in-app tap would. This decides WHERE, never WHETHER.
 */

/** A router target, deliberately not a raw path string. */
export interface DeepLinkTarget {
  name: 'player' | 'podcast' | 'topic' | 'person'
  params: Record<string, string>
  /** `?t=<seconds>` passed through, so a link can name a MOMENT and not just an episode. */
  query?: Record<string, string>
}

/** Ids we mint are slugs and feed ids; anything else is not ours. */
const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/

/**
 * `<host>` → the route it means. Both the singular and plural host are accepted because links get
 * written by hand and by other systems, and a 404 on a shared link is a bad way to learn about a
 * naming preference.
 */
const TARGETS: Record<string, { name: DeepLinkTarget['name']; param: string }> = {
  episode: { name: 'player', param: 'slug' },
  episodes: { name: 'player', param: 'slug' },
  podcast: { name: 'podcast', param: 'feedId' },
  show: { name: 'podcast', param: 'feedId' },
  topic: { name: 'topic', param: 'id' },
  person: { name: 'person', param: 'id' },
}

export const APP_SCHEME = 'closelistening'

/**
 * The route an inbound URL asks for, or null when it asks for nothing we serve.
 *
 * Accepts both `closelistening://episode/<slug>` (custom scheme, where the "host" is the target)
 * and `https://<any-host>/episode/<slug>` (a universal link, or simply the web URL someone
 * copied), so one parser covers however the link arrived.
 */
export function routeForDeepLink(raw: string): DeepLinkTarget | null {
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    return null
  }

  // The auth callback is NOT a navigation, and must never be treated as one — it carries a token
  // in the fragment and is handled by the auth listener.
  if (url.host === 'auth' || url.pathname.startsWith('/auth')) return null

  const segments = [url.host, ...url.pathname.split('/')].map((s) => s.trim()).filter(Boolean)
  // Custom scheme: host IS the target ("closelistening://episode/x" → ['episode','x']). Web URL:
  // the host is a domain, so the target is the first PATH segment — try both readings, in that
  // order, and take the first that names something we serve.
  for (const [head, id] of [
    [segments[0], segments[1]],
    [segments[1], segments[2]],
  ]) {
    const target = head ? TARGETS[head.toLowerCase()] : undefined
    if (!target || !id) continue
    const decoded = safeDecode(id)
    if (!decoded || !ID_PATTERN.test(decoded)) return null
    return { name: target.name, params: { [target.param]: decoded }, ...startTimeOf(url) }
  }
  return null
}

/**
 * `{ query: { t } }` when the link named a usable start time, otherwise nothing.
 *
 * Written as an explicit null/empty check rather than `Number(...)` alone, because `Number(null)`
 * and `Number('')` are both **0** — so an absent `t` silently became "start at zero" and every
 * link claimed a moment it had not named.
 *
 * Validated here rather than at the router: an inbound URL is attacker-supplied, and a NaN
 * reaching `el.currentTime` throws. An unusable value is dropped rather than refusing the link —
 * losing the moment is a shame, losing the episode is a broken link.
 */
function startTimeOf(url: URL): { query: Record<string, string> } | Record<string, never> {
  const raw = url.searchParams.get('t')
  if (raw === null || raw.trim() === '') return {}
  const seconds = Number(raw)
  if (!Number.isFinite(seconds) || seconds < 0) return {}
  return { query: { t: String(Math.floor(seconds)) } }
}

/** A malformed percent-escape throws; an unusable id is not a reason to crash the handler. */
function safeDecode(value: string): string | null {
  try {
    return decodeURIComponent(value)
  } catch {
    return null
  }
}
