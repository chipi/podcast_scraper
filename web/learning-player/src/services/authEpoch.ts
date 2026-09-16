/**
 * Remembers the server's session-key fingerprint so a MASS invalidation can be told apart from an
 * ordinary expired session (incident 2026-09-16).
 *
 * The problem this solves: a 401 is ambiguous. It means "your credential is not valid", which is
 * true both when a single user's token has aged out AND when the server has lost or rotated its
 * signing key and just invalidated every token on the platform at once. The client's response has
 * to differ:
 *
 *   - my session expired      → sign out, clear the cached content for that account. Correct.
 *   - the server rotated keys → the user did nothing wrong and their library is still theirs.
 *                               Keep the cache, ask them to sign in again.
 *
 * Treating the second as the first is what made the production incident destructive rather than
 * merely annoying. `/api/health` now publishes `auth_epoch`; this records the last one seen and
 * reports when it changes.
 *
 * Deliberately NOT in `deviceStore`: this is a fact about a SERVER, not about the person using the
 * device, and it must survive a sign-out (that is exactly when it is needed next).
 */

const EPOCH_KEY = 'lp.authEpoch'

function read(): string | null {
  try {
    return localStorage.getItem(EPOCH_KEY)
  } catch {
    return null
  }
}

/**
 * Record the epoch reported by the server; returns `true` when it DIFFERS from the one last seen.
 *
 * A first sighting (nothing stored yet) is not a rotation — a fresh install has no prior key to
 * have been rotated away from, and claiming otherwise would make every first launch look like an
 * incident. `null` (auth unconfigured) is ignored entirely rather than stored, so a deployment that
 * turns auth off and on again does not read as a rotation either.
 */
export function noteAuthEpoch(epoch: string | null | undefined): boolean {
  if (!epoch) return false
  const previous = read()
  if (previous === epoch) return false
  try {
    localStorage.setItem(EPOCH_KEY, epoch)
  } catch {
    /* storage blocked — we simply cannot detect rotation on this device; never fail a launch */
  }
  return previous !== null
}

/** Forget the recorded epoch (used when the user signs out deliberately). */
export function clearAuthEpoch(): void {
  try {
    localStorage.removeItem(EPOCH_KEY)
  } catch {
    /* ignore */
  }
}
