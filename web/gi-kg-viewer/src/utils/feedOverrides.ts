import type { FeedApiEntry } from '../api/feedsApi'

/**
 * Helpers for the per-feed override editor (#694). A feed entry is either a bare
 * URL string or an object with a `url`/`rss` key plus optional override fields
 * (validated server-side against `RssFeedEntry`). These helpers split an entry
 * into editable parts and recombine them, enforcing two rules:
 *
 *  - **empty value = field absent** (never persist `max_episodes: null` etc.), and
 *  - **collapse to a plain string** when no overrides/extras remain.
 *
 * Unknown keys (advanced tuning the editor doesn't surface as inputs) are kept
 * verbatim in `extras` so nothing is silently dropped.
 */
export type FeedUrlKey = 'url' | 'rss'
export type EpisodeOrder = 'newest' | 'oldest'

export interface FeedMustFields {
  max_episodes?: number
  episode_order?: EpisodeOrder
  episode_offset?: number
  episode_since?: string
  episode_until?: string
}

/** The five "must" override fields surfaced as structured inputs (#694). */
export const FEED_MUST_FIELD_KEYS = [
  'max_episodes',
  'episode_order',
  'episode_offset',
  'episode_since',
  'episode_until',
] as const

/** Rarer per-feed tuning fields, surfaced as structured inputs in the
 *  collapsible Advanced block (#694). Types drive input rendering + coercion. */
export type FeedAdvancedFieldType = 'int' | 'float' | 'bool' | 'string'
export interface FeedAdvancedFieldDef {
  key: string
  label: string
  type: FeedAdvancedFieldType
  group: string
}
export const FEED_ADVANCED_FIELDS: FeedAdvancedFieldDef[] = [
  { key: 'timeout', label: 'Timeout (s)', type: 'float', group: 'HTTP / RSS retry' },
  { key: 'http_retry_total', label: 'HTTP retry total', type: 'int', group: 'HTTP / RSS retry' },
  { key: 'http_backoff_factor', label: 'HTTP backoff factor', type: 'float', group: 'HTTP / RSS retry' },
  { key: 'rss_retry_total', label: 'RSS retry total', type: 'int', group: 'HTTP / RSS retry' },
  { key: 'rss_backoff_factor', label: 'RSS backoff factor', type: 'float', group: 'HTTP / RSS retry' },
  { key: 'delay_ms', label: 'Delay (ms)', type: 'int', group: 'Delay / concurrency' },
  { key: 'host_request_interval_ms', label: 'Host request interval (ms)', type: 'int', group: 'Delay / concurrency' },
  { key: 'host_max_concurrent', label: 'Host max concurrent', type: 'int', group: 'Delay / concurrency' },
  { key: 'circuit_breaker_enabled', label: 'Circuit breaker enabled', type: 'bool', group: 'Circuit breaker' },
  { key: 'circuit_breaker_failure_threshold', label: 'Failure threshold', type: 'int', group: 'Circuit breaker' },
  { key: 'circuit_breaker_window_seconds', label: 'Window (s)', type: 'int', group: 'Circuit breaker' },
  { key: 'circuit_breaker_cooldown_seconds', label: 'Cooldown (s)', type: 'int', group: 'Circuit breaker' },
  { key: 'circuit_breaker_scope', label: 'Scope', type: 'string', group: 'Circuit breaker' },
  { key: 'rss_conditional_get', label: 'RSS conditional GET', type: 'bool', group: 'Conditional GET' },
  { key: 'rss_cache_dir', label: 'RSS cache dir', type: 'string', group: 'Conditional GET' },
  { key: 'episode_retry_max', label: 'Episode retry max', type: 'int', group: 'Episode retry' },
  { key: 'episode_retry_delay_sec', label: 'Episode retry delay (s)', type: 'float', group: 'Episode retry' },
  { key: 'user_agent', label: 'User agent', type: 'string', group: 'Misc' },
]
export const FEED_ADVANCED_FIELD_KEYS = FEED_ADVANCED_FIELDS.map((f) => f.key)

export interface SplitFeedEntry {
  urlKey: FeedUrlKey
  url: string
  must: FeedMustFields
  /** Known advanced fields present on the entry (rendered as inputs). */
  advanced: Record<string, unknown>
  /** Keys the editor doesn't model — preserved verbatim via the raw-JSON box. */
  extras: Record<string, unknown>
}

export function splitFeedEntry(entry: FeedApiEntry): SplitFeedEntry {
  if (typeof entry === 'string') {
    return { urlKey: 'url', url: entry.trim(), must: {}, advanced: {}, extras: {} }
  }
  const o = (entry ?? {}) as Record<string, unknown>
  const urlKey: FeedUrlKey = o.url !== undefined ? 'url' : o.rss !== undefined ? 'rss' : 'url'
  const url = String(o.url ?? o.rss ?? '').trim()
  const must: FeedMustFields = {}
  if (typeof o.max_episodes === 'number') must.max_episodes = o.max_episodes
  if (o.episode_order === 'newest' || o.episode_order === 'oldest') {
    must.episode_order = o.episode_order
  }
  if (typeof o.episode_offset === 'number') must.episode_offset = o.episode_offset
  if (typeof o.episode_since === 'string') must.episode_since = o.episode_since
  if (typeof o.episode_until === 'string') must.episode_until = o.episode_until
  const skip = new Set<string>(['url', 'rss', ...FEED_MUST_FIELD_KEYS])
  const advancedKeys = new Set<string>(FEED_ADVANCED_FIELD_KEYS)
  const advanced: Record<string, unknown> = {}
  const extras: Record<string, unknown> = {}
  for (const k of Object.keys(o)) {
    if (skip.has(k)) continue
    if (advancedKeys.has(k)) advanced[k] = o[k]
    else extras[k] = o[k]
  }
  return { urlKey, url, must, advanced, extras }
}

export function buildFeedEntry(
  urlKey: FeedUrlKey,
  url: string,
  must: FeedMustFields,
  extras: Record<string, unknown> = {},
): FeedApiEntry {
  const obj: Record<string, unknown> = {}
  obj[urlKey] = url.trim()
  for (const [k, v] of Object.entries(extras)) {
    obj[k] = v
  }
  // Must-fields win over any same-named extra; omit empties (= inherit).
  if (must.max_episodes != null) obj.max_episodes = must.max_episodes
  if (must.episode_order) obj.episode_order = must.episode_order
  if (must.episode_offset != null) obj.episode_offset = must.episode_offset
  if (must.episode_since) obj.episode_since = must.episode_since
  if (must.episode_until) obj.episode_until = must.episode_until
  const keys = Object.keys(obj)
  if (keys.length === 1 && keys[0] === urlKey) {
    return url.trim()
  }
  return obj as FeedApiEntry
}

/** True when the entry carries any non-URL key (override or advanced tuning). */
export function feedEntryHasOverrides(entry: FeedApiEntry): boolean {
  if (typeof entry === 'string') return false
  const o = (entry ?? {}) as Record<string, unknown>
  return Object.keys(o).some((k) => k !== 'url' && k !== 'rss')
}
