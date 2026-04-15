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
  try {
    return await fetch(input, {
      ...init,
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch (e) {
    if (isAbortOrTimeout(e)) {
      const suffix = detail ? ` (${detail})` : ''
      throw new Error(`Request timed out after ${timeoutMs}ms${suffix}`)
    }
    throw e
  }
}
