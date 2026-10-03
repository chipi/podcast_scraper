/**
 * Which distribution channel this build came from (#2265).
 *
 * ── Why this is baked and not detected ───────────────────────────────────────
 * A TestFlight build and an App Store build are the SAME code, on the same platform, reporting the
 * same `platform: 'ios'`. Nothing in the running app can tell them apart. The only moment the
 * difference is known is when the build is produced, so the producing lane stamps it in via the
 * `APP_CHANNEL` env var → `__APP_CHANNEL__` (see vite.config.ts).
 *
 * ── Why it matters ───────────────────────────────────────────────────────────
 * It is the whole basis of the saved Umami view `channel in (testflight, play_internal)`, which is
 * what separates the beta cohort's sessions from the operator's own usage and from web traffic.
 * Every per-person report depends on that separation.
 *
 * ── The honest-unknown rule ──────────────────────────────────────────────────
 * No release lane sets `APP_CHANNEL` yet — #2189 (iOS → TestFlight / App Store) and #2191/#2192
 * (Android → Play internal / production) are the issues that must, and none of those lanes exists
 * today. So until then a native build has no declared channel, and this resolves it to `unknown`
 * instead of guessing.
 *
 * Both guesses available here are worse than admitting ignorance:
 *   - guessing `web` for a native build is a plainly false value, and it makes the beta filter
 *     return nothing while looking like it is working;
 *   - guessing `testflight` would quietly fold the operator's own device into the beta cohort.
 * `unknown` is visible in the data, so a missing channel shows up as a question rather than as a
 * confident wrong answer. It is an addition to the spec's six values, for exactly that reason.
 */

import { Capacitor } from '@capacitor/core'

/** The spec's six channels, plus `unknown` for a native build whose lane did not stamp one. */
export const CHANNELS = [
  'testflight',
  'play_internal',
  'app_store',
  'play_store',
  'web',
  'dev',
  'unknown',
] as const

export type Channel = (typeof CHANNELS)[number]

function isDeclared(value: string): value is Channel {
  return (CHANNELS as readonly string[]).includes(value)
}

/**
 * Resolve the channel for this build.
 *
 * A stamped value wins, but only if it is one this app knows: a typo in a release lane
 * (`APP_CHANNEL=testfilght`) must not become a silent new category in the dashboards, so an
 * unrecognised value is treated the same as an absent one.
 */
export function resolveChannel(): Channel {
  const stamped = typeof __APP_CHANNEL__ === 'string' ? __APP_CHANNEL__.trim() : ''
  if (stamped && isDeclared(stamped)) return stamped

  // Nothing stamped (or something unrecognised). On a native shell we genuinely do not know.
  if (Capacitor.isNativePlatform()) return 'unknown'

  // On the web we do: a dev server is `dev`, anything else is the deployed web app.
  return import.meta.env.DEV ? 'dev' : 'web'
}
