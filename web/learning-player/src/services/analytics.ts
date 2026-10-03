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
 * ── Script injection lives here, and only here ───────────────────────────────
 * `installUmami()` is the single injection path; `main.ts` calls it. It used to be an inline block
 * in `main.ts`, and it moved because sign-out needs to REPLACE the tracker (see `resetIdentity`) —
 * two copies of the injection logic would have drifted the moment one of them gained the
 * `data-exclude-search` attribute and the other did not. It is idempotent: a second call with a
 * tag already present does nothing, because two tags double-count every page view.
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

import { Capacitor } from '@capacitor/core'
import { resolveChannel, type Channel } from './channel'

// ── Session properties ───────────────────────────────────────────────────────

/**
 * Set once, through `identify`, and attached to every event in the session.
 *
 * Deliberately short. The spec is explicit that locale, device model and anything Umami already
 * derives must NOT be sent, and that the beta cohort must NOT be stamped on events — cohorts are
 * computed at read time from the operator's `analytics_id` list, so that re-running a report with
 * a corrected list does not require re-collecting the data.
 */
export type SessionProps = {
  platform: 'ios' | 'android' | 'web'
  app_version: string
  channel: Channel
}

/**
 * Build the session properties for this build and device.
 *
 * Reads `Capacitor.getPlatform()` directly rather than `services/native`'s `platform()` wrapper,
 * and that matters more than it looks: `native.ts` calls `registerPlugin()` at module load, so
 * importing it here dragged the whole native plugin layer into EVERY module that tracks an event.
 * `useOnline` then blew up two unrelated test files whose `@capacitor/core` mock had no
 * `registerPlugin` — a telemetry helper should not be able to do that.
 */
export function resolveSession(): SessionProps {
  const p = Capacitor.getPlatform()
  return {
    platform: p === 'ios' || p === 'android' ? p : 'web',
    app_version: typeof __APP_VERSION__ === 'string' ? __APP_VERSION__ : '',
    channel: resolveChannel(),
  }
}

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
  /**
   * No `is_new_account`, and that is a correction to the spec rather than an omission.
   *
   * The client cannot compute it honestly. The only client-side signal available is "did this
   * device already hold an auth snapshot", which answers a different question: a tester
   * reinstalling, or signing in on a second phone, would be reported as a signup. That error runs
   * in the worst direction for the beta, inflating exactly the activation numbers it exists to
   * measure.
   *
   * The spec already provides the right source for this: the SERVER emits `account_created`
   * (#2266), explicitly as "server-side truth for signups, independent of the Umami script
   * loading". Duplicating it here with a worse signal would add noise to a number that is already
   * correct elsewhere.
   */
  auth_completed: { provider: string }
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
  /**
   * Widened beyond the spec's `shows | topics | people`, because that set can express neither half
   * of the real Browse surface (2026-10-03).
   *
   * The Browse hub's own tabs are **Episodes** and **Shows**; the trends section inside it switches
   * between **topics**, **storylines** and **people**. So `episodes` and `storylines` had no value
   * to report, and an event that cannot name the surface a listener chose answers nothing. The
   * question it exists for — "which browse surface do people actually use" — is the same for both
   * rows of controls, so both report through here.
   */
  browse_tab_view: { tab: 'episodes' | 'shows' | 'topics' | 'people' | 'storylines' }
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
  /**
   * `density_tick` removed (2026-10-03): there is no such opener.
   *
   * The spec offers `button | density_tick`, but the panel has exactly one opener — the pill in
   * PlayerView — and `EpisodeDensity` emits `seek`, not a panel open. An enum value that can never
   * be emitted is the same failure as a diagnostic that looks like it is recording something: it
   * makes a dashboard look like it is answering "how do people get in" when only one answer was
   * ever possible. Add it back together with the affordance, if one is built.
   */
  knowledge_panel_open: { trigger: 'button' }
  insight_tap: { insight_type: string }
  transcript_seek: undefined
  /** NEVER the text of the highlight or note. */
  capture_created: {
    kind: 'highlight' | 'note'
    /**
     * Widened beyond the spec's four with `show` and `storyline` (2026-10-03).
     *
     * `NoteTarget` in services/types.ts genuinely includes both, and folding a note written
     * against a SHOW into `episode` would be a wrong value rather than a coarse one — the two are
     * different acts, and "they annotate shows" is a finding the beta would want. `highlight`
     * folds into `episode` because a note on a highlight IS a note on a moment of that episode.
     */
    target_kind: 'episode' | 'topic' | 'person' | 'insight' | 'show' | 'storyline'
  }
  recap_view: { trigger: 'panel' | 'home_prompt' }
  revisit_open: { item_kind: string }
  queue_add: { source: Source }
  collection_add: undefined
  download_start: undefined
  share: {
    /**
     * `organization` added beyond the spec's five (2026-10-03). `EntityCardBody` renders
     * person / topic / ORGANIZATION, and its share menu is the same component — so without this,
     * an org share had to be filed as a topic. A wrong kind is worse than a new one: it would make
     * organizations invisible while inflating topics.
     */
    target_kind: 'episode' | 'moment' | 'topic' | 'person' | 'storyline' | 'organization'
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

/**
 * NOTHING IS HARDCODED HERE, and that is the point (operator, 2026-10-03).
 *
 * This block used to carry a dev script URL and a dev website id as literals. Both were WRONG, and
 * the way they were wrong is the argument against ever hardcoding them: the id
 * `30384fd4-b22b-406c-b5f6-054a0e0d16d1` does not exist in the Umami instance. Measured by posting
 * it to `/api/send`:
 *
 *     {"error":{"message":"Website not found.","code":"bad-request","status":400}}
 *
 * So every event sent from `vite dev` was rejected — silently, because `track()` is fire-and-forget
 * and the tracker swallows its own errors. Dev analytics looked wired and went nowhere, which is
 * indistinguishable from "nobody used the app". (The same literal in the operator viewer's
 * `lib/analytics.ts` is equally dead; it is not this arc's file to change.)
 *
 * Both values now come from the environment, identically in every tier:
 *
 *   - `VITE_UMAMI_SRC`        — the full tracking-script URL, injected verbatim, never suffixed
 *   - `VITE_UMAMI_WEBSITE_ID` — the site to report to
 *
 * Prod bakes them as docker build-args; local dev puts them in `web/learning-player/.env.local`
 * (gitignored). `VITE_UMAMI_SRC_DEV` stays supported for the on-device dev tier, where `.env.mobile`
 * supplies an https URL because a native WebView blocks the plain-http one as mixed content.
 *
 * With neither set, analytics is a true no-op — the fork-silent default. A build that forgets them
 * sends nothing, which is the correct failure: better silent than reporting into the wrong site.
 */
/**
 * The hard kill switch, honoured ahead of every env value (#2264 §2.6).
 *
 * This nearly went missing. Removing the hardcoded dev defaults also removed `devDefaultEnabled()`,
 * which held the ONLY reference to `VITE_ANALYTICS_OFF` — so for one commit the switch did nothing
 * and a `.env.local` would have made the unit suite send real events. Checked here, in front of
 * everything, so no combination of env values can get past it.
 */
function analyticsOff(): boolean {
  return import.meta.env.VITE_ANALYTICS_OFF === '1'
}

function umamiSrc(): string {
  if (analyticsOff()) return ''
  return (
    (import.meta.env.VITE_UMAMI_SRC as string) ||
    (import.meta.env.VITE_UMAMI_SRC_DEV as string) ||
    ''
  )
}

function umamiWebsiteId(): string {
  if (analyticsOff()) return ''
  return (import.meta.env.VITE_UMAMI_WEBSITE_ID as string) || ''
}

/**
 * Record the user's choice from Settings › Privacy (#2265).
 *
 * Writes Umami's own `umami.disabled` flag rather than a key of our own, so the tracker honours it
 * internally too — belt and braces: even a `track()` call added later that forgot the gate stays
 * silent. Verified in the served `script.js`: its enablement check reads
 * `g?.getItem("umami.disabled")`.
 *
 * Note the asymmetry the spec requires: this silences UMAMI only. Server-side listen events keep
 * flowing, because they power the user's own listening stats in the app — switching those off would
 * take away a feature the person can see, not just telemetry they cannot.
 */
export function setUserOptedOut(optedOut: boolean): void {
  try {
    if (optedOut) localStorage.setItem('umami.disabled', '1')
    else localStorage.removeItem('umami.disabled')
  } catch {
    // Blocked storage: the preference cannot be persisted, so leave it as it was rather than
    // throwing inside a settings toggle.
  }
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
  identify?: (id: string | Record<string, unknown>, props?: Record<string, unknown>) => void
}

/** Marks our tag so injection is idempotent and `resetIdentity` can find it again. */
const TAG_MARKER = 'data-umami-installed'

/**
 * The id currently attached to this tracker, or `null` when anonymous.
 *
 * Exists so `identify` is once-per-id (see there) and so `resetIdentity` can make a later
 * re-identify with the SAME id go through again — after a tracker replacement the new tracker knows
 * nothing, so suppressing that call would leave the session permanently anonymous.
 */
let identifiedAs: string | null = null

/**
 * Inject the Umami `<script>` exactly once. Idempotent and safe to call before the app mounts.
 *
 * `data-exclude-search` is set here, and it is the whole reason the search term stays out of
 * analytics: Umami auto-tracks the full URL, and five call sites put the term in the query string
 * (`SearchView.vue:357,438,501`, `BrowseView.vue:45`, `HomeView.vue:317`, `LibraryView.vue:636`).
 * Verified supported on the running Umami 3.3.1 by reading the served `script.js`
 * (`j = w("exclude-search") === b`), not the docs.
 */
export function installUmami(): void {
  try {
    if (typeof document === 'undefined') return
    const src = umamiSrc()
    const websiteId = umamiWebsiteId()
    // No script url or no website id → nothing was ever going to work; stay silent by construction
    // (a fork that checks out the repo and builds sends nothing).
    if (!src || !websiteId) return
    if (document.querySelector(`script[${TAG_MARKER}]`)) return
    const tag = document.createElement('script')
    tag.defer = true
    tag.src = src
    tag.setAttribute('data-website-id', websiteId)
    tag.setAttribute('data-exclude-search', 'true')
    tag.setAttribute(TAG_MARKER, '1')
    document.head.appendChild(tag)
  } catch {
    // APPENDING A SCRIPT CAN THROW, and this runs during app bootstrap in `main.ts` — before the
    // Pinia/router wiring — so an uncaught throw here takes the whole app down over a metric.
    //
    // Not hypothetical: happy-dom raises `DOMException [NotSupportedError]: JavaScript file loading
    // is disabled` from `appendChild`, which is how this was found. A restrictive CSP or a
    // locked-down WebView can do the same in production. The inline block this replaced had the
    // identical exposure and no guard.
  }
}

/**
 * Attach every subsequent event to this account's pseudonymous id (#2265).
 *
 * An empty id is ignored rather than sent: an account whose backfill has not run yet has none, and
 * identifying with a blank string would create one shared bucket that looks like a single very
 * busy participant.
 */
export function identify(analyticsId: string, session: SessionProps): void {
  try {
    if (!analyticsId) return
    if (!analyticsEnabled()) return
    if (typeof window === 'undefined') return
    // ONCE PER ID. The auth store calls this from `refresh()`, which runs on every boot and every
    // background revalidation, and each `identify` is a network payload. Re-sending the same id
    // would put a request on the wire every time the app regained focus for no new information.
    if (identifiedAs === analyticsId) return
    const umami = (window as unknown as { umami?: UmamiGlobal }).umami
    umami?.identify?.(analyticsId, { ...session })
    identifiedAs = analyticsId
  } catch {
    // Fire-and-forget, as everywhere else here.
  }
}

/**
 * Stop attributing events to the account that just signed out.
 *
 * REPLACES the tracker, because it cannot be asked to forget. Measured in the served `script.js`:
 * `identify` derives the id as `typeof t === "string" ? t : t.id` and then assigns only
 * `void 0 !== a && (V = a)` — so `identify({})` leaves the previous distinct id in place. The
 * script also guards its own global with `t.umami || (t.umami = {...})`, so merely re-injecting the
 * tag is a no-op. The global has to go first, then the tag, then a fresh install.
 *
 * Why it matters: sign-out here is client-side only (`stores/auth.ts` sets `user = null`, no
 * reload), so without this the signed-out landing traffic the spec wants to be an ANONYMOUS
 * session would keep carrying the previous participant's id — and in a beta where the operator and
 * a tester may share a device, that is one person's browsing filed under another's name.
 */
export function resetIdentity(): void {
  try {
    if (typeof window !== 'undefined') {
      delete (window as unknown as { umami?: UmamiGlobal }).umami
    }
    if (typeof document !== 'undefined') {
      document.querySelectorAll(`script[${TAG_MARKER}]`).forEach((el) => el.remove())
    }
    identifiedAs = null
    installUmami()
  } catch {
    // A failed reset must not break sign-out. The cost is attribution, not access.
  }
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
