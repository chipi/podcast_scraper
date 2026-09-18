import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Web-only delivery APIs must have a native branch (operator 2026-09-18).
 *
 * ## The class of bug this exists for
 *
 * Three separate features were dead on the installed iOS build, each for the same reason and each
 * shipped green: the build passes, the type-check passes, and 1500 unit tests pass, because nothing
 * in the suite runs in a WKWebView.
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
})
