/**
 * #1075 chunk 4 — RFC-094 OQ-1 panel cache for the relational layer.
 *
 * Today every `subject.focusPerson()` re-fires the four relational
 * endpoints (`/api/relational/topics`, `/api/relational/co-speakers`,
 * `/api/relational/positions`, CIL `/api/persons/{id}/brief`) — even
 * when the user just toggled the Person Profile tab back to a Person
 * they viewed two seconds ago. Tab-switching feels laggier than the
 * data justifies.
 *
 * This composable wraps each of the four fetch functions with a
 * module-level LRU + TTL cache keyed by `(corpusPath, fn, args...)`.
 * Repeat focus within the TTL window returns the cached payload
 * synchronously (modulo `await` semantics) — the tab feels instant.
 *
 * Invalidation:
 *   - Cache keys include `corpusPath`, so switching corpus naturally
 *     produces a different key space — old-corpus entries never bleed
 *     into new-corpus queries (they sit until LRU eviction or TTL
 *     expiry, but are never returned).
 *   - The TTL is short (30 s) so any backend mutation feeds back to
 *     the viewer within human-scale wait.
 *   - `invalidateRelationalCache()` is exported for tests + as the
 *     explicit "things may have changed under us" hook.
 *
 * Scope:
 *   - Wraps `fetchPersonTopics`, `fetchCoSpeakers`, `fetchPositions`
 *     (from `relationalApi.ts`) and `fetchPersonProfile` (from
 *     `cilApi.ts`). These are the four endpoints PersonLandingView
 *     fires on every Person focus.
 *   - Same call signature as the wrapped fn so consumers can swap in
 *     place. Returns an unmodified `Promise<T>` on hit OR miss.
 */
import { fetchPersonProfile as _fetchPersonProfile } from '../api/cilApi'
import {
  fetchCoSpeakers as _fetchCoSpeakers,
  fetchPersonTopics as _fetchPersonTopics,
  fetchPositions as _fetchPositions,
} from '../api/relationalApi'

/** Default TTL — 30 s feels instant for tab-switching, slow enough that
 *  background mutations propagate back within a human-scale wait. */
const DEFAULT_TTL_MS = 30_000
/** LRU bound. The viewer focuses a handful of Persons per session;
 *  32 entries comfortably covers a heavy hop-around without growing
 *  unbounded. */
const DEFAULT_MAX_ENTRIES = 32

interface CacheEntry<T> {
  ts: number
  value: T
}

/** Module-level cache singleton. Tests reset via
 *  ``invalidateRelationalCache()``. */
const cache = new Map<string, CacheEntry<unknown>>()

let ttlMs: number = DEFAULT_TTL_MS
let maxEntries: number = DEFAULT_MAX_ENTRIES

/** Build a deterministic cache key. Joined with a control-character
 *  separator so an arg containing `::` can't collide with another
 *  arg's value. */
function makeKey(corpusPath: string, fn: string, args: string[]): string {
  return `${corpusPath}${fn}${args.join('')}`
}

function getEntry<T>(key: string): T | undefined {
  const entry = cache.get(key)
  if (!entry) return undefined
  if (Date.now() - entry.ts > ttlMs) {
    cache.delete(key)
    return undefined
  }
  // LRU touch — re-insert moves to the end.
  cache.delete(key)
  cache.set(key, entry)
  return entry.value as T
}

function setEntry(key: string, value: unknown): void {
  if (cache.size >= maxEntries) {
    // Drop the oldest (Map iteration is insertion order).
    const oldest = cache.keys().next().value as string | undefined
    if (oldest !== undefined) cache.delete(oldest)
  }
  cache.set(key, { ts: Date.now(), value })
}

/** Clear the entire relational cache. Exported as the manual hook for
 *  tests + future explicit invalidation needs (e.g. a settings page
 *  forcing a refresh after a config change). */
export function invalidateRelationalCache(): void {
  cache.clear()
}

/** Test-only helpers (no-op for runtime callers). */
export function _setCacheParamsForTest(opts: { ttlMs?: number; maxEntries?: number }): void {
  if (opts.ttlMs !== undefined) ttlMs = opts.ttlMs
  if (opts.maxEntries !== undefined) maxEntries = opts.maxEntries
}
export function _resetCacheParamsForTest(): void {
  ttlMs = DEFAULT_TTL_MS
  maxEntries = DEFAULT_MAX_ENTRIES
}
export function _cacheSizeForTest(): number {
  return cache.size
}

// Note on corpus-switch invalidation: the cache key includes
// `corpusPath`, so switching corpus naturally produces a different key
// space — old corpus entries hang around until the LRU evicts them or
// the TTL expires, but they are never returned to a new-corpus query.
// Explicit operator-triggered invalidation lives at
// ``invalidateRelationalCache``; consumers that need eager cache flush
// on corpus change call it directly (e.g. from a settings page).

/** Cached wrapper for `fetchPersonTopics(corpus, person, k?)`. */
export async function cachedFetchPersonTopics(
  corpusPath: string,
  personId: string,
  k?: number,
): ReturnType<typeof _fetchPersonTopics> {
  const key = makeKey(corpusPath, 'fetchPersonTopics', [personId, String(k ?? '')])
  const hit = getEntry<Awaited<ReturnType<typeof _fetchPersonTopics>>>(key)
  if (hit !== undefined) return hit
  // Omit ``k`` from the inner call when undefined so the downstream
  // mock spy in PersonLandingView.test.ts sees the same 2-arg shape
  // the un-cached path used.
  const result = k === undefined
    ? await _fetchPersonTopics(corpusPath, personId)
    : await _fetchPersonTopics(corpusPath, personId, k)
  setEntry(key, result)
  return result
}

/** Cached wrapper for `fetchCoSpeakers(corpus, person, k?)`. */
export async function cachedFetchCoSpeakers(
  corpusPath: string,
  personId: string,
  k?: number,
): ReturnType<typeof _fetchCoSpeakers> {
  const key = makeKey(corpusPath, 'fetchCoSpeakers', [personId, String(k ?? '')])
  const hit = getEntry<Awaited<ReturnType<typeof _fetchCoSpeakers>>>(key)
  if (hit !== undefined) return hit
  const result = k === undefined
    ? await _fetchCoSpeakers(corpusPath, personId)
    : await _fetchCoSpeakers(corpusPath, personId, k)
  setEntry(key, result)
  return result
}

/** Cached wrapper for `fetchPositions(corpus, person, k?)`. */
export async function cachedFetchPositions(
  corpusPath: string,
  personId: string,
  k?: number,
): ReturnType<typeof _fetchPositions> {
  const key = makeKey(corpusPath, 'fetchPositions', [personId, String(k ?? '')])
  const hit = getEntry<Awaited<ReturnType<typeof _fetchPositions>>>(key)
  if (hit !== undefined) return hit
  const result = k === undefined
    ? await _fetchPositions(corpusPath, personId)
    : await _fetchPositions(corpusPath, personId, k)
  setEntry(key, result)
  return result
}

/** Cached wrapper for `fetchPersonProfile(corpus, person)` (CIL brief). */
export async function cachedFetchPersonProfile(
  corpusPath: string,
  personId: string,
): ReturnType<typeof _fetchPersonProfile> {
  const key = makeKey(corpusPath, 'fetchPersonProfile', [personId])
  const hit = getEntry<Awaited<ReturnType<typeof _fetchPersonProfile>>>(key)
  if (hit !== undefined) return hit
  const result = await _fetchPersonProfile(corpusPath, personId)
  setEntry(key, result)
  return result
}
