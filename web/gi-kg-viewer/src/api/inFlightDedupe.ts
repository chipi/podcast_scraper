/**
 * Coalesce concurrent identical logical requests into one in-flight promise.
 *
 * This is a **cost** optimization only: it does not cache responses after completion.
 * A second call after the first finishes always runs a new request.
 */
const inFlight = new Map<string, Promise<unknown>>()

export function dedupeInFlight<T>(key: string, run: () => Promise<T>): Promise<T> {
  const hit = inFlight.get(key)
  if (hit) {
    return hit as Promise<T>
  }
  const pending = (async (): Promise<T> => {
    try {
      return await run()
    } finally {
      inFlight.delete(key)
    }
  })()
  inFlight.set(key, pending)
  return pending
}
