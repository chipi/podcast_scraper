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

/** The cached value, or null. Never throws — a cache miss and a broken cache are the same thing. */
export async function readCached<T>(key: string): Promise<T | null> {
  try {
    if (!isNative()) return await getDeviceJson<T>(deviceKey(key))
    const { data } = await Filesystem.readFile({
      path: filePath(key),
      directory: CACHE_DIR,
      encoding: Encoding.UTF8,
    })
    return JSON.parse(typeof data === 'string' ? data : '') as T
  } catch {
    return null
  }
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
export const CACHE_KEYS = ['library', 'favorites', 'queue'] as const
