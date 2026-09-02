/**
 * Client-minted capture ids (#1925).
 *
 * Capture is append-only, so a POST whose RESPONSE was lost — the ordinary offline case, since the
 * write may well have landed — could only be retried by risking a duplicate. That is why
 * highlights and notes were the one thing kept OUT of the offline outbox. A key the client picks
 * before the request removes the ambiguity: the server stores the first write and returns it
 * unchanged for every replay.
 *
 * The shape has to match the server's `_CLIENT_ID_PATTERN` (`[A-Za-z0-9_-]{1,64}`).
 */

/**
 * `crypto.randomUUID` is not available on every surface this runs on (older iOS WKWebView, and
 * insecure-origin dev servers), and a capture that throws because of an id generator would be a
 * self-inflicted outage. `getRandomValues` is far more widely present; the last resort is only
 * reached when neither exists.
 */
function randomToken(): string {
  const c = globalThis.crypto
  if (c?.randomUUID) return c.randomUUID().replace(/-/g, '')
  if (c?.getRandomValues) {
    const bytes = c.getRandomValues(new Uint8Array(16))
    return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
  }
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 12)}`
}

/** A fresh id for a capture. `prefix` is cosmetic — it makes a log line readable. */
export function newCaptureId(prefix: 'h' | 'n'): string {
  return `${prefix}c_${randomToken()}`
}
