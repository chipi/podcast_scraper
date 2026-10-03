/**
 * Umami custom-event tracking for the consumer player (#2264, epic #2263).
 *
 * Mirrors the operator viewer's `web/gi-kg-viewer/src/lib/analytics.ts`, with one deliberate
 * difference: props are typed PER EVENT instead of `Record<string, unknown>`. The beta analytics
 * spec requires that every property be an enum or a bucket, with no free text ever — no search
 * terms, no note or highlight text, no names the user typed. Across the 39 events of #2267 that
 * cannot be held by review, so it is held by the compiler: an unlisted event name, a misspelled
 * enum value or a raw string where a bucket belongs is a type error.
 *
 * ── Script injection is NOT here ─────────────────────────────────────────────
 * The `<script>` tag is injected in `main.ts`, which also sets `data-exclude-search` to keep the
 * search term out of tracked URLs. Do not add a second injection path here: two tags would
 * double-count every page view, and the viewer's `initAnalytics()` exists only because the viewer
 * has no equivalent block in its own `main.ts`.
 *
 * ── Enablement ───────────────────────────────────────────────────────────────
 * Gated on the same pair `main.ts` gates the script on, so `track()` cannot be live while the
 * script is absent. `VITE_ANALYTICS_OFF=1` hard-disables the dev default, and the vitest config
 * sets it so unit runs never emit.
 *
 * The user's own opt-out is Umami's `umami.disabled` localStorage flag (#2265). Umami's script
 * honours it internally, so `track()` would already be inert — it is checked here as well so the
 * no-op is explicit and testable rather than a property of a third-party bundle.
 *
 * ── Never break the app ──────────────────────────────────────────────────────
 * Telemetry is fire-and-forget: every call is wrapped, and a throw inside the tracker is
 * swallowed. A failed metric must never surface to a listener.
 */

// ── Shared vocabularies ──────────────────────────────────────────────────────

/**
 * Where a navigation came from. This is the most important property in the spec: it is what makes
 * "do people move across the corpus, or just play shows they already follow?" answerable, and it
 * is the input to Discovery share, Pivot rate and the derived cross-show hop.
 */
export const SOURCES = [
  'home_your_week',
  'home_key_voices',
  'home_momentum',
  'home_trending_topics',
  'home_trending_shows',
  'home_storylines',
  'browse',
  'search',
  'player',
  'knowledge_panel',
  'entity_card',
  'entity_page',
  'storyline_page',
  'library',
  'queue',
  'revisit',
  'recap',
  'notification',
  'deep_link',
  'other',
] as const
export type Source = (typeof SOURCES)[number]

/** Counts are reported as buckets, never exact. */
export type CountBucket = '0' | '1' | '2-5' | '6-20' | '21+'
/** Positions in a list are reported as buckets, never exact. */
export type RankBucket = '1' | '2-3' | '4-10' | '11+'
/** Durations are reported as buckets, never exact. */
export type DurationBucket = '<1m' | '1-5m' | '5-15m' | '15-45m' | '45m+'

/**
 * Bucket a count.
 *
 * Exported and tested because the alternative is each call site inlining its own thresholds, which
 * is how two events end up disagreeing about what "a few" means and a dashboard silently compares
 * incomparable numbers. Negative and non-finite inputs collapse to '0' rather than throwing — a
 * bucketing helper must not be able to break a render.
 */
export function toCountBucket(n: number): CountBucket {
  if (!Number.isFinite(n) || n <= 0) return '0'
  if (n === 1) return '1'
  if (n <= 5) return '2-5'
  if (n <= 20) return '6-20'
  return '21+'
}

/** Bucket a 1-based position in a list. Anything below 1 is treated as 1. */
export function toRankBucket(n: number): RankBucket {
  if (!Number.isFinite(n) || n <= 1) return '1'
  if (n <= 3) return '2-3'
  if (n <= 10) return '4-10'
  return '11+'
}

/** Bucket a duration given in SECONDS. */
export function toDurationBucket(seconds: number): DurationBucket {
  if (!Number.isFinite(seconds) || seconds < 60) return '<1m'
  if (seconds < 300) return '1-5m'
  if (seconds < 900) return '5-15m'
  if (seconds < 2700) return '15-45m'
  return '45m+'
}

// ── The event registry ───────────────────────────────────────────────────────

/**
 * The canonical player event vocabulary. `track()` accepts only these names.
 *
 * All 39 are listed up front, before their call sites exist (#2267 wires those), because the names
 * and their props ARE the contract: the dashboards, the funnel and every metric are defined in
 * terms of them. Freezing them here means the call-site work cannot quietly rename or re-shape one.
 *
 * `show_missing` is deliberately absent — the spec qualifies it as needing a "can't find" affordance
 * "if and when it exists", and no such affordance does. Its signal is carried instead by
 * `empty_state_shown` with `reason: 'not_in_corpus'`.
 */
export const EVENT_NAMES = [
  // page views
  'screen_view',
  // onboarding and activation
  'landing_view',
  'landing_cta_click',
  'landing_teaser_click',
  'auth_started',
  'auth_completed',
  'auth_failed',
  'interests_picker_shown',
  'interests_saved',
  'interests_dismissed',
  // discovery and pivots
  'home_rail_click',
  'browse_tab_view',
  'entity_open',
  'episode_open',
  'follow',
  'unfollow',
  'search_submitted',
  'search_result_click',
  'share_link_opened',
  // listening and learning
  'play_start',
  'knowledge_panel_open',
  'insight_tap',
  'transcript_seek',
  'capture_created',
  'recap_view',
  'revisit_open',
  'queue_add',
  'collection_add',
  'download_start',
  'share',
  'route_output',
  'highlights_export',
  'speed_change',
  // friction and health
  'error_shown',
  'empty_state_shown',
  'offline_session',
  'update_prompt',
  // notifications
  'push_permission',
  'notification_open',
] as const

export type EventName = (typeof EVENT_NAMES)[number]

/** Kinds of thing a user can follow, open or capture against. */
export type EntityKind = 'topic' | 'person' | 'show' | 'storyline'

/**
 * The props each event carries. `undefined` means the event takes none.
 *
 * Every value is a literal union or a bucket. There is deliberately no `string` anywhere in this
 * map except `screen` (a route name, which is developer-authored and low-cardinality) and
 * `insight_type` / `item_kind` / `target_kind` / `format` / `provider` / `surface`, which are
 * corpus or platform vocabularies rather than anything the user types.
 */
export type EventProps = {
  screen_view: { screen: string }

  landing_view: undefined
  landing_cta_click: { cta: 'create_account' | 'sign_in'; position: 'hero' | 'closing' }
  landing_teaser_click: { kind: 'show' | 'topic' }
  auth_started: { provider: string }
  auth_completed: { provider: string; is_new_account: boolean }
  auth_failed: { provider: string; reason: 'cancelled' | 'error' }
  interests_picker_shown: { trigger: 'first_run' | 'profile' | 'home_prompt' }
  interests_saved: { count: CountBucket }
  interests_dismissed: undefined

  home_rail_click: {
    rail:
      | 'your_week'
      | 'key_voices'
      | 'momentum'
      | 'trending_topics'
      | 'trending_shows'
      | 'storylines'
    rank: RankBucket
  }
  browse_tab_view: { tab: 'shows' | 'topics' | 'people' }
  entity_open: { kind: EntityKind; presentation: 'card' | 'page'; source: Source }
  episode_open: { source: Source; from_followed_show: boolean }
  follow: { kind: 'show' | 'person' | 'topic'; source: Source }
  unfollow: { kind: 'show' | 'person' | 'topic'; source: Source }
  /** NO query text. `results` is a bucket, so "how many hits" never becomes "what they searched". */
  search_submitted: { scope: 'corpus' | 'recall'; results: CountBucket }
  search_result_click: {
    result_kind: 'episode' | 'topic' | 'person' | 'show' | 'insight'
    rank: RankBucket
  }
  share_link_opened: { target_kind: string }

  play_start: { surface: 'player' | 'mini_player' | 'queue'; resumed: boolean }
  knowledge_panel_open: { trigger: 'button' | 'density_tick' }
  insight_tap: { insight_type: string }
  transcript_seek: undefined
  /** NEVER the text of the highlight or note. */
  capture_created: {
    kind: 'highlight' | 'note'
    target_kind: 'episode' | 'topic' | 'person' | 'insight'
  }
  recap_view: { trigger: 'panel' | 'home_prompt' }
  revisit_open: { item_kind: string }
  queue_add: { source: Source }
  collection_add: undefined
  download_start: undefined
  share: {
    target_kind: 'episode' | 'moment' | 'topic' | 'person' | 'storyline'
    method: 'native_sheet' | 'copy_link'
  }
  /**
   * No `destination`, and that is a correction to the spec rather than an omission.
   *
   * The spec gives this event a `destination` "enum from RouteButton". That value cannot be
   * obtained: `components/RouteButton.vue` records that tapping opens the PLATFORM's own device
   * sheet (AirPlay on iOS, the Cast/output picker on Android), that the choice is made outside the
   * app, and that no API exists for a page — "or even a native app" — to enumerate AirPlay / Cast /
   * Bluetooth targets. So this records only that the user reached for output routing, which is all
   * the app can honestly observe. Do not synthesise a destination.
   */
  route_output: undefined
  highlights_export: { format: string }
  speed_change: { speed: '1' | '1.25' | '1.5' | '1.75' | '2+' }

  error_shown: { surface: string; kind: 'network' | 'server' | 'not_found' | 'auth' }
  empty_state_shown: { surface: string; reason: 'no_results' | 'no_content' | 'not_in_corpus' }
  offline_session: undefined
  update_prompt: { kind: 'pwa_toast' | 'native_banner'; action: 'accepted' | 'dismissed' }

  push_permission: { result: 'granted' | 'denied' | 'deferred' }
  notification_open: { type: string; channel: 'push' | 'bell' }
}

// ── Enablement ───────────────────────────────────────────────────────────────

const DEV_UMAMI_SRC = 'http://homelab:3001/script.js'
const DEV_UMAMI_WEBSITE_ID = '30384fd4-b22b-406c-b5f6-054a0e0d16d1'

/** Dev default is live in `vite dev` unless a runner explicitly opted out. */
function devDefaultEnabled(): boolean {
  return import.meta.env.DEV && import.meta.env.VITE_ANALYTICS_OFF !== '1'
}

function umamiSrc(): string {
  return (
    (import.meta.env.VITE_UMAMI_SRC as string) ||
    (import.meta.env.VITE_UMAMI_SRC_DEV as string) ||
    (devDefaultEnabled() ? DEV_UMAMI_SRC : '')
  )
}

function umamiWebsiteId(): string {
  return (
    (import.meta.env.VITE_UMAMI_WEBSITE_ID as string) ||
    (devDefaultEnabled() ? DEV_UMAMI_WEBSITE_ID : '')
  )
}

/** True when the user has turned analytics off in Settings (#2265). */
export function userOptedOut(): boolean {
  try {
    return localStorage.getItem('umami.disabled') === '1'
  } catch {
    // Private mode, blocked storage, a thrown accessor. Absence of a flag is not consent to
    // ignore it, but a storage error must not decide the question either way — treat it as
    // "not opted out" and let Umami's own check be the backstop.
    return false
  }
}

/**
 * Whether a `track()` call will do anything.
 *
 * Mirrors the gate in `main.ts`: no script URL or no website id means no script was injected, so
 * tracking cannot work and must not pretend to.
 */
export function analyticsEnabled(): boolean {
  return !!umamiSrc() && !!umamiWebsiteId() && !userOptedOut()
}

type UmamiGlobal = {
  track?: (name: string, props?: Record<string, unknown>) => void
}

/**
 * Record one custom event.
 *
 * The name is constrained to {@link EVENT_NAMES} and the props to {@link EventProps}, so a typo or
 * an out-of-vocabulary value is a compile error rather than a silently-wrong dashboard.
 *
 * Safe to call before the Umami script has loaded (it queues internally) and a no-op when
 * analytics is disabled or the user opted out. Never throws.
 */
export function track<N extends EventName>(
  name: N,
  ...args: EventProps[N] extends undefined ? [] : [props: EventProps[N]]
): void {
  try {
    if (!analyticsEnabled()) return
    if (typeof window === 'undefined') return
    const umami = (window as unknown as { umami?: UmamiGlobal }).umami
    umami?.track?.(name, args[0] as Record<string, unknown> | undefined)
  } catch {
    // Fire-and-forget: a metric that fails must never reach the listener.
  }
}
