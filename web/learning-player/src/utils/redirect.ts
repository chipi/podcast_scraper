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
