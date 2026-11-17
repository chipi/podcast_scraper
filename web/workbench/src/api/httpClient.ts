/**
 * Shared viewer HTTP client: timeouts on all API fetches to avoid hung spinners.
 */

export const DEFAULT_VIEWER_FETCH_TIMEOUT_MS = 120_000

export type FetchWithTimeoutOptions = {
  timeoutMs?: number
  /** Appended to timeout error for debugging (e.g. relative artifact path). */
  timeoutDetail?: string
}

export function isAbortOrTimeout(e: unknown): boolean {
  if (e instanceof DOMException && e.name === 'AbortError') {
    return true
  }
  return e instanceof Error && e.name === 'AbortError'
}

/**
 * ``fetch`` with ``AbortSignal.timeout``. Maps abort to a readable ``Error``.
 */
export async function fetchWithTimeout(
  input: RequestInfo | URL,
  init?: RequestInit,
  opts?: FetchWithTimeoutOptions,
): Promise<Response> {
  const timeoutMs = opts?.timeoutMs ?? DEFAULT_VIEWER_FETCH_TIMEOUT_MS
  const detail = opts?.timeoutDetail?.trim()
  /* Compose the built-in timeout signal with any caller-supplied signal
   * (e.g. hydrate()'s 5 s AbortController). AbortSignal.any resolves on
   * whichever fires first — dropping the caller's signal (as the old
   * ``signal: AbortSignal.timeout(...)`` did) would silently defeat any
   * client-side deadline the caller set. */
  const timeoutSignal = AbortSignal.timeout(timeoutMs)
  const composedSignal = init?.signal
    ? AbortSignal.any([init.signal, timeoutSignal])
    : timeoutSignal
  try {
    return await fetch(input, {
      ...init,
      // Send the shared session cookie (lp_session) so the viewer reuses the player's auth
      // (#1128). Same-origin in prod; the dev proxy forwards cookies to :8000.
      credentials: init?.credentials ?? 'include',
      signal: composedSignal,
    })
  } catch (e) {
    if (isAbortOrTimeout(e)) {
      const suffix = detail ? ` (${detail})` : ''
      throw new Error(`Request timed out after ${timeoutMs}ms${suffix}`)
    }
    throw e
  }
}
