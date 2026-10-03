/**
 * Query-string scrubbing for error telemetry (#2264).
 *
 * The player puts the user's search term in the URL (`/search?q=<term>`), from five call sites.
 * Umami's `data-exclude-search` keeps it out of analytics page views, but Sentry/GlitchTip is a
 * SECOND sink for the same text, and `sendDefaultPii: false` does not cover it — that option
 * governs IP address, cookies and user data, not query strings.
 *
 * Traced through the installed SDK (@sentry/vue 10.60.0) rather than assumed:
 *
 *   - `@sentry/browser/.../integrations/breadcrumbs.js:229-234` sets a navigation breadcrumb's
 *     `from` / `to` to `parseUrl(...).relative`.
 *   - `@sentry/core/src/utils/url.ts` defines that field as `path + query + fragment`
 *     ("everything minus origin").
 *
 * So a navigation to `/search?q=<term>` is recorded verbatim, and breadcrumbs attach to every
 * error event. For contrast — so the gap reads as a gap and not as Sentry being careless — the
 * tracing path IS sanitised: span names go through `getSanitizedUrlStringFromUrlObject`, which
 * clears `search` and `hash`, and the browser tracing integration names navigations from
 * `location.pathname`. Breadcrumbs are the one path that keeps the query string.
 *
 * This lives in its own module, separate from the `Sentry.init` call, for one reason: a hook
 * defined inline in `main.ts` can only be checked by asserting that some text appears in the
 * source, which proves the hook exists and nothing about whether it strips anything. Here the
 * behaviour is testable directly.
 */

/** A navigation breadcrumb, narrowed to the fields this module touches. */
export type ScrubbableBreadcrumb = {
  category?: string
  data?: Record<string, unknown>
}

/**
 * Drop the query string and fragment from a URL or path.
 *
 * Splits on the FIRST `?` or `#`, so a term that itself contains either character cannot smuggle
 * part of itself past the cut. A value that is not a string is returned untouched — breadcrumb
 * data is `unknown`, and a scrubber that throws would take the error report down with it.
 */
export function stripQuery(value: unknown): unknown {
  return typeof value === 'string' ? value.split(/[?#]/, 1)[0] : value
}

/**
 * Scrub a breadcrumb in place and return it, for use as Sentry's `beforeBreadcrumb`.
 *
 * Only `category: 'navigation'` is touched. Other categories are left exactly as they are: this
 * is a targeted fix for a measured leak, not a blanket rewrite of telemetry, and silently
 * altering unrelated breadcrumbs would make future debugging harder for no privacy gain.
 */
export function scrubNavigationBreadcrumb<T extends ScrubbableBreadcrumb>(breadcrumb: T): T {
  if (breadcrumb.category === 'navigation' && breadcrumb.data) {
    breadcrumb.data.from = stripQuery(breadcrumb.data.from)
    breadcrumb.data.to = stripQuery(breadcrumb.data.to)
  }
  return breadcrumb
}

/** An event, narrowed to the one field this module touches. */
export type ScrubbableEvent = { request?: { url?: string } }

/**
 * Scrub `request.url` on an outgoing event, for use as Sentry's `beforeSend`.
 *
 * Defensive rather than measured: I did not establish where the browser SDK populates
 * `event.request.url`, only that `@sentry/core/.../prepareEvent.js:67-69` truncates it when it is
 * present and never strips its query. Rather than assume it is always unset, strip it if it is
 * there — one `split` against the chance of leaking the same text by a second route.
 */
export function scrubEventRequestUrl<T extends ScrubbableEvent>(event: T): T {
  if (event.request?.url) {
    event.request.url = stripQuery(event.request.url) as string
  }
  return event
}
