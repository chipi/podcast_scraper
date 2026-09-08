/**
 * Per-account content cache (#1909) — "everything I loaded last time is still there, just stale."
 *
 * Deliberately NOT the service worker. There is none on iOS (observed on a simulator, #1908),
 * `CapacitorHttp` bypasses it anyway, and the native API is a different origin. And per-user
 * endpoints are excluded from the SW cache on purpose: Cache Storage is keyed by URL and global to
 * the origin, so after a sign-out it would happily serve one account's data to the next. This
 * cache is namespaced by `user_id` and only ever read for the identity that owns it.
 *
 * Governing rule, shared with auth and PlayerView:
 * **only a 401/403 may destroy cached state; a transport error never may.**
 *
 * Storage: Filesystem JSON on native (episode lists and rails are far too big for
 * UserDefaults/SharedPreferences, which load the whole store into memory); device KV on web,
 * where payloads are small and the SW already covers shared GETs.
 */

import { Directory, Encoding, Filesystem } from '@capacitor/filesystem'
import { getDeviceJson, removeDeviceKey, setDeviceJson } from './deviceStore'
import { isNative } from './native'

const CACHE_DIR = Directory.LibraryNoCloud
const CACHE_FOLDER = 'content-cache'
export const ANON_NAMESPACE = 'anon'

let namespace = ANON_NAMESPACE

/** Injected by the shell when identity resolves or changes, like the downloads registry. */
export function setCacheNamespace(ns: string): void {
  namespace = ns || ANON_NAMESPACE
}

export function cacheNamespace(): string {
  return namespace
}

function safe(key: string): string {
  return key.replace(/[^a-zA-Z0-9._-]/g, '_')
}

function filePath(key: string): string {
  return `${CACHE_FOLDER}/${safe(namespace)}/${safe(key)}.json`
}

function deviceKey(key: string): string {
  return `cache.${namespace}.${key}`
}

/**
 * The cached value, or null. Never throws — a cache miss and a broken cache are the same thing.
 *
 * ## Why the validator
 *
 * `as T` was a promise the data never made. Anything that has ever been written under a key stays
 * on the device: a shape from an older app version, a half-written file, a value some future
 * refactor renames. The reader handed all of it back as `T`, and the callers put it straight into
 * a store — so a stale-format entry did not degrade a list, it made the view RENDER-THROW on the
 * `.length` of something that was no longer an array. Observed while testing Library: a cached
 * favourites entry of the wrong shape took the whole tab down.
 *
 * That is the one failure mode this cache exists to prevent. The arc's rule is that a failed read
 * must never destroy what the user has; a read that crashes the view is worse than a failed one.
 *
 * `isValid` is a cheap shape check, not a schema — enough to answer "could this still be a T". A
 * value that fails it is treated as a MISS, which is exactly right: we have no usable copy, so the
 * caller should fetch, and next write replaces the bad entry.
 */
export async function readCached<T>(
  key: string,
  isValid?: (value: unknown) => boolean,
): Promise<T | null> {
  try {
    const raw = !isNative()
      ? await getDeviceJson<unknown>(deviceKey(key))
      : JSON.parse(
          ((await Filesystem.readFile({
            path: filePath(key),
            directory: CACHE_DIR,
            encoding: Encoding.UTF8,
          }).then((r) => r.data)) as string) || '',
        )
    if (raw === null || raw === undefined) return null
    if (isValid && !isValid(raw)) return null
    return raw as T
  } catch {
    return null
  }
}

/** The commonest guard: the key holds a list. */
export function isArrayCache(value: unknown): boolean {
  return Array.isArray(value)
}

/**
 * The next commonest: an object carrying the named keys as arrays.
 *
 * No `!Array.isArray` clause. It reads as though it were rejecting a cached list handed to an
 * object guard, but `JSON.parse` cannot produce an array with named properties, so it could never
 * fire — and a plain array fails the field check on its own. An unreachable clause that looks
 * defensive is worse than none: it implies a hazard that does not exist here.
 */
export function hasArrayFields(...fields: string[]): (value: unknown) => boolean {
  return (value: unknown): boolean =>
    typeof value === 'object' &&
    value !== null &&
    fields.every((f) => Array.isArray((value as Record<string, unknown>)[f]))
}

/** Never throws: failing to cache is not a reason to fail the read that produced the data. */
export async function writeCached(key: string, value: unknown): Promise<void> {
  try {
    if (!isNative()) {
      await setDeviceJson(deviceKey(key), value)
      return
    }
    await Filesystem.writeFile({
      path: filePath(key),
      directory: CACHE_DIR,
      data: JSON.stringify(value),
      encoding: Encoding.UTF8,
      recursive: true,
    })
  } catch {
    // Disk full, or the folder is gone. The live data is already in memory.
  }
}

/**
 * Drop one account's cache. Called on sign-out and on a 401 — the two cases where the cached
 * content is genuinely no longer ours to show.
 */
export async function clearCached(keys: readonly string[]): Promise<void> {
  for (const key of keys) {
    try {
      if (!isNative()) await removeDeviceKey(deviceKey(key))
      else await Filesystem.deleteFile({ path: filePath(key), directory: CACHE_DIR })
    } catch {
      // Already gone.
    }
  }
}

/** The keys the app caches, so sign-out can clear all of them without hunting. */
export const CACHE_KEYS = [
  'library',
  'favorites',
  'queue',
  'collections',
  // The Home rails (#1909). They were in the issue's scope from the start and were the half that
  // never landed, which is why Home was a column of "Couldn't load this right now" with no network.
  // Listed here so sign-out clears them with everything else — a rail is per-account content too,
  // and leaving one behind would show the previous user's Home to the next one.
  'home.whatsnew',
  'home.catalogue',
  'home.continue',
  'home.recommended',
  'home.yourweek',
  'home.storylines',
  'home.trendingtopics',
  'home.trendingshows',
  // The player's per-episode snapshots (#16/#1909). Per-account content like any other:
  // leaving them behind would paint the previous user's episode page for the next one.
  'player.snapshots',
  'browse.episodes',
  'captures',
  // Browse tabs. The trending ones are per WINDOW: switching to 6m offline must not blank a 3m
  // list we actually have.
  'browse.topics.1m',
  'browse.topics.3m',
  'browse.topics.6m',
  'browse.topics.1y',
  'browse.people.1m',
  'browse.people.3m',
  'browse.people.6m',
  'browse.people.1y',
  'browse.storylines',
  'browse.shows',
] as const
