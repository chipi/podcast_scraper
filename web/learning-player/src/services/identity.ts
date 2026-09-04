/**
 * One identity epoch, shared by every per-account store (#1925 review C11/C12).
 *
 * The arc kept re-deriving the same rule per module and then shipping the next module without it:
 * an in-flight load that resolves AFTER an account switch must not write its result into the new
 * account's state. The downloads store grew a private `loadGeneration` for exactly this; the queue,
 * library and favourites stores did not, so account A's in-flight GET could land in B's store and
 * B's cache file.
 *
 * Rather than a fourth private copy, this is the single counter. Bump it whenever the signed-in
 * identity changes; capture it before an await and re-check after.
 */

let epoch = 0

export function identityEpoch(): number {
  return epoch
}

export function bumpIdentityEpoch(): number {
  epoch += 1
  return epoch
}

/** True when the identity changed while the caller was awaiting — abandon the result. */
export function identityChangedSince(captured: number): boolean {
  return captured !== epoch
}

/** Test seam. */
export function __resetIdentityEpoch(): void {
  epoch = 0
}
