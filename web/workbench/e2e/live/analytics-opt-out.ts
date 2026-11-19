/**
 * Keep the live smoke out of PRODUCTION analytics.
 *
 * The live smoke browses the real site, so its page views landed in the production Umami website:
 * ~100 per deploy on 2026-10-06, the GitHub runner's tests all counted as one "session" (same IP +
 * UA) and read as a burst of repeat page views. Umami's tracker skips any browser whose
 * localStorage carries `umami.disabled` (checked in the deployed `script.js`), so every context the
 * smoke opens starts with it set. No app code involved.
 *
 * Contexts made with `browser.newContext()` do NOT inherit the config's `use`, so pass this there
 * too.
 */
export function umamiOptOut(origin: string): {
  cookies: []
  origins: { origin: string; localStorage: { name: string; value: string }[] }[]
} {
  return {
    cookies: [],
    origins: [{ origin: new URL(origin).origin, localStorage: [{ name: 'umami.disabled', value: '1' }] }],
  }
}
