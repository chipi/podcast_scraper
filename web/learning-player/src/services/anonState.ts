/**
 * Wipe the anonymous scratch namespace (#1925 review).
 *
 * Every device-local store falls back to `anon` when nobody is signed in: playback positions, the
 * listen queue, the outbox, and the content cache. On a shared device that made signing out a
 * LEAK — user A signs out, their positions stay under `anon`, and whoever picks the phone up next
 * sees A's "continue listening" before they have signed in as anyone.
 *
 * Only a TRANSITION to signed-out purges. Booting straight into the signed-out app must not, or a
 * listener who has never had an account loses their position on every launch — they have no server
 * copy to restore it from, so that is the one place this data is the only copy there is.
 */

import { ANON_NAMESPACE, CACHE_KEYS, cacheNamespace, clearCached, setCacheNamespace } from './contentCache'
import { removeDeviceKey } from './deviceStore'
import { PENDING_KEY_PREFIX, __resetListenLog } from './listenLog'
import { OUTBOX_KEY_PREFIX, __resetOutbox } from './outbox'
import { POSITIONS_KEY_PREFIX, __resetPositions } from './playbackPositions'

/** Every device-local key that is namespaced. Add yours here when you add a store. */
const NAMESPACED_PREFIXES = [PENDING_KEY_PREFIX, OUTBOX_KEY_PREFIX, POSITIONS_KEY_PREFIX]

export async function purgeAnonymousState(): Promise<void> {
  __resetPositions()
  __resetListenLog()
  __resetOutbox()
  for (const prefix of NAMESPACED_PREFIXES) {
    await removeDeviceKey(`${prefix}.${ANON_NAMESPACE}`)
  }
  // clearCached works on the CURRENT namespace, so point it at anon and put it back — the caller
  // is mid-switch and the namespace it wants is not necessarily the one set right now.
  const before = cacheNamespace()
  setCacheNamespace(ANON_NAMESPACE)
  await clearCached(CACHE_KEYS)
  setCacheNamespace(before)
}
