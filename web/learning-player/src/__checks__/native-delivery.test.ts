import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Web-only delivery APIs must have a native branch (operator 2026-09-18).
 *
 * ## The class of bug this exists for
 *
 * Three separate features were dead on the installed iOS build, each for the same reason and each
 * shipped green: the build passes, the type-check passes, and 1500 unit tests pass — none of which
 * runs in a WKWebView.
 *
 * I first wrote here that "nothing in the suite runs in a WKWebView". That was false, and the
 * operator caught it. `ios/uitests/` holds a full XCUITest suite, and `NativeCapabilityTests`
 * covers the share sheet specifically — the very mechanism these exports use. What is true is
 * narrower and more damning: that suite had NO make target, so it had never run. A test nobody
 * invokes is indistinguishable from a test nobody wrote, which is exactly the trap of asserting
 * absence instead of checking.
 *
 * - `<a download>` saves nothing in WKWebView. The Obsidian export hit this, and the "fix" was to
 *   hide the button on native — so the feature simply vanished on the phone rather than appearing
 *   broken. The episode-notes Markdown chip hit the same thing and was not hidden, so it was a
 *   button that did nothing.
 * - `window.open(url, '_blank')` is a silent no-op in WKWebView: nothing opens, nothing throws. Both
 *   PDF exports (highlights and episode notes) were this.
 *
 * Both failures are SILENT. There is no console error, no rejected promise, no 4xx — the tap simply
 * does nothing, which is indistinguishable from the feature not existing. That is why they survived
 * to a device: every automated signal we have says the code is fine.
 *
 * ## Why the check is "is there a native branch" rather than "is it correct"
 *
 * This cannot verify the native path WORKS — only a device can. What it can do is refuse the shape
 * that produced all three defects: a web-only delivery API with no `isNative()` anywhere near it.
 * The failure mode is forgetting the branch exists, not writing a wrong one.
 *
 * Hiding the control on native does NOT satisfy this. That was the Obsidian "fix", and it is worse
 * than the broken button: a missing feature reads as a product decision, so it goes unreported.
 */

const SRC = join(__dirname, '..')

/** APIs that do nothing in the native WebView. */
const WEB_ONLY = [
  { pattern: /\.download\s*=/, name: '<a download>' },
  { pattern: /:download=/, name: '<a :download>' },
  { pattern: /window\.open\(/, name: 'window.open' },
  { pattern: /window\.print\(/, name: 'window.print' },
]

/**
 * Files exempt because they ARE the native branch. Add nothing here without a reason that survives
 * being read aloud.
 */
const EXEMPT = new Set([
  // The shared helper — `openExternal` is the branch (it reaches window.open only on web).
  'native.ts',
])

/**
 * Comments are not code.
 *
 * Both assertions below fired on prose: the fix for the Obsidian defect DOCUMENTS the old
 * `v-if="!isNative()"` so the next reader knows why the branch exists, and that sentence read as a
 * fresh violation. A guard that punishes you for explaining the bug it exists to catch teaches
 * people to delete the explanation.
 */
function stripComments(text: string): string {
  return text
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1')
}

function sources(dir: string): { name: string; path: string; text: string }[] {
  const out: { name: string; path: string; text: string }[] = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name)
    if (entry.isDirectory()) {
      if (entry.name === '__checks__' || entry.name === '__tests__') continue
      out.push(...sources(p))
    } else if (/\.(vue|ts)$/.test(entry.name) && !/\.(test|spec)\.ts$/.test(entry.name)) {
      out.push({ name: entry.name, path: p, text: stripComments(readFileSync(p, 'utf8')) })
    }
  }
  return out
}

describe('native delivery (operator 2026-09-18)', () => {
  it('no web-only download/open API ships without a native branch', () => {
    const offenders: string[] = []
    for (const f of sources(SRC)) {
      if (EXEMPT.has(f.name)) continue
      const hits = WEB_ONLY.filter((w) => w.pattern.test(f.text))
      if (!hits.length) continue
      // A native branch anywhere in the file: the guarded helper, or an explicit platform check.
      const branched = /isNative\(\)|isNativePlatform\(\)|openExternal\(/.test(f.text)
      if (!branched) offenders.push(`${f.name} → ${hits.map((h) => h.name).join(', ')}`)
    }

    expect(
      offenders,
      `These use a delivery API that does NOTHING in the iOS/Android WebView, with no native ` +
        `branch in the file. The tap fails SILENTLY — no error, no rejection — so it reads to the ` +
        `user as the feature not existing, and no test we run can see it. Route through ` +
        `openExternal() (for window.open) or saveAndShareText/saveAndShareBinary (for downloads). ` +
        `Hiding the control on native is NOT a fix: that is how the Obsidian export disappeared.`,
    ).toEqual([])
  })

  it('no feature is hidden from the native build instead of being made to work', () => {
    // `v-if="!isNative()"` means "this exists on the web but not on your phone". The Obsidian
    // export used it to paper over a delivery bug, and the operator reported the feature as
    // "disappeared". Each new one must be justified here rather than added silently.
    const hidden = sources(SRC)
      .filter((f) => f.name.endsWith('.vue'))
      .filter((f) => /v-if="!isNative\(\)"/.test(f.text))
      .map((f) => f.name)

    expect(
      hidden,
      `These hide a control on native. The Obsidian export did exactly this — its zip was ` +
        `delivered by <a download>, which WKWebView ignores — and the whole feature vanished on ` +
        `the phone. Fix the delivery instead. If a control is genuinely web-only, say why in a ` +
        `comment and add it to this assertion deliberately.`,
    ).toEqual([])
  })

  /**
   * A SECOND class, found on the device after the first was fixed (operator 2026-09-19).
   *
   * `openExternal` satisfies the check above — it is the native branch. But it opens
   * SFSafariViewController, which does NOT share the app's WKWebView cookie jar. Hand it an
   * AUTHENTICATED url and the request arrives signed-out: the export route returned the sign-in
   * gate instead of the document, and the operator's report carried the tell — "when I copy the
   * link from there and open it in a normal browser, it works fine", because that browser had a
   * session.
   *
   * This failure is not silent, which is why it took a person rather than a test to find: something
   * DOES open, it just shows the wrong page. It reads as a rendering bug rather than an auth one.
   *
   * So: an `/api/app/*` url may not be handed to `openExternal` on the native path. Fetch it with
   * the app's credentials and share the bytes — the pattern every other export already uses.
   */
  it('no authenticated API url is handed to an external browser on native', () => {
    const offenders: string[] = []
    for (const f of sources(SRC)) {
      if (EXEMPT.has(f.name)) continue
      for (const m of f.text.matchAll(/openExternal\(([^)]*)\)/g)) {
        const arg = m[1]
        // The url-building helpers for auth-gated export routes. A literal `/api/app/` path counts
        // too — the point is the route, not how the string was assembled.
        const authed = /Url\(|\/api\/app\//.test(arg)
        if (!authed) continue
        // Acceptable only when the call is guarded by a web-only branch, i.e. native takes the
        // fetch-and-share path instead.
        const guarded = /if\s*\(!isNative\(\)\)/.test(f.text)
        if (!guarded) offenders.push(`${f.name} → openExternal(${arg.trim()})`)
      }
    }

    expect(
      offenders,
      `These hand an authenticated /api/app url to openExternal, which opens an external browser ` +
        `with NO session cookie — so the user sees the sign-in gate instead of their document. ` +
        `Fetch it with credentials and pass the bytes to saveAndShareText, the way the Markdown ` +
        `and Obsidian exports already do, and keep openExternal for the web branch only.`,
    ).toEqual([])
  })
})
