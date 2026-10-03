/**
 * RFC-120 (#2009): a post-login redirect target we're willing to follow must be a **same-origin
 * absolute path**. Rejects protocol-relative (`//host`, `/\host`), absolute URLs, and control
 * characters — mirrors the backend's `_safe_return_to` open-redirect guard (app_auth.py). Returns
 * the safe path, or null. Used by the router guard, LoginView, and LandingView so the login-first
 * `?redirect` funnel can't be turned into an open redirect.
 */
export function safeInternalPath(value: unknown): string | null {
  if (typeof value !== 'string') return null
  if (!value.startsWith('/')) return null
  // `//host` and `/\host` are protocol-relative — the browser would navigate off-origin.
  if (value.startsWith('//') || value.startsWith('/\\')) return null
  // Reject C0 control characters (CR/LF etc.).
  for (let i = 0; i < value.length; i++) {
    if (value.charCodeAt(i) < 0x20) return null
  }
  return value
}

/** The two public pages a signed-out person waits on. Neither means anything once signed in. */
const SIGNED_OUT_PAGES = new Set(['landing', 'login'])

/**
 * Where a sign-in that arrived by LINK (native deep link: OAuth on Android, the email magic link on
 * both platforms) should take the app, or null to stay put.
 *
 * - A just-created account goes to its profile (`welcome=1`), exactly as the web verify redirect
 *   lands it: an email identity arrives with no name and no picture.
 * - Otherwise, only a signed-out page is left: to its `?redirect` target if safe, else home. The
 *   link can be opened with the app anywhere, and most often it was closed in the meantime and
 *   booted to `/welcome` — which has no sign-in watch of its own (only `/login` does), so the person
 *   sat signed in under a "Create your free account" page (measured on the simulator 2026-10-03).
 * - Anywhere else, the person was already somewhere they chose; leave them there.
 */
export function postLinkSignInRoute(
  isNew: boolean,
  current: { name?: unknown; query?: Record<string, unknown> },
): { name: string; query?: Record<string, string> } | { path: string } | null {
  if (isNew) return { name: 'profile', query: { welcome: '1' } }
  if (typeof current.name !== 'string' || !SIGNED_OUT_PAGES.has(current.name)) return null
  const target = safeInternalPath(current.query?.redirect)
  return target ? { path: target } : { name: 'home' }
}
