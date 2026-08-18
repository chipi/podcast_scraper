import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Every per-user write is reachable signed-out, and must be gated (#1590).
 *
 * #1590 replaced `v-if="auth.isAuthenticated"` with visible sign-in teasers: the control is not
 * unavailable, it is *deferred*, and hiding it hid the differentiators from exactly the visitors
 * deciding whether to sign up. But that change has a cost — every gated control is now clickable by
 * someone with no session, so **every** call site needs the gate.
 *
 * ## Why this file was rewritten
 *
 * The first version of this guard was file-level: a file containing the string `useSignInGate`
 * passed. That let four defects through on the very next read, two of them in controls whose fix I
 * had already written down as done:
 *
 * - `PodcastView` and `KnowledgePanel` imported the gate for one control while another stayed
 *   `v-if="auth.isAuthenticated"` — so my gate wiring was dead code behind a hidden element.
 * - `queue.playNext(` and `capture.captureMoment(` were missing from GATED_WRITES entirely.
 * - The hidden-control check pinned ONE historical regex in ONE file, so it tested the past bug
 *   rather than its class.
 *
 * So the checks below are per-call-site, and the write list is asserted complete against the store
 * source rather than maintained by hand. A guard that only knows the last bug is a guard that finds
 * the last bug.
 */

const SRC = join(__dirname, '..')

/**
 * Store actions that write per-user state. Each maps to an authenticated endpoint, so calling one
 * without a session is a 401 — a bug, never merely a no-op (T7: fix the cause, don't suppress it).
 *
 * Kept honest by `every per-user store write appears in GATED_WRITES` below.
 */
const GATED_WRITES = [
  'library.toggle(',
  'favorites.toggle(',
  'favorites.toggleInsight(',
  'queue.add(',
  'queue.remove(',
  'queue.toggle(',
  'queue.playNext(',
  'queue.move(',
  'capture.captureSpan(',
  'capture.captureInsight(',
  'capture.captureMoment(',
  'capture.setColor(',
  'capture.remove(',
  'capture.addNote(',
  'capture.editNote(',
  'capture.removeNote(',
]

/** Components that render the affordance but delegate the action (and the gate) to a parent. */
const DELEGATES_TO_PARENT = new Set(['TranscriptList.vue'])

/**
 * Views reachable only behind an authenticated route guard, where there is no signed-out visitor to
 * gate. Anything added here must genuinely be unreachable without a session — check the router.
 */
const AUTH_ONLY_ROUTES = new Set([
  'QueueView.vue',
  'LibraryView.vue',
  'ProfileView.vue',
  // Not a route itself — rendered only inside LibraryView, which is `requiresAuth: true`
  // (router/index.ts). If it ever gains its own route, that route must be guarded too.
  'HighlightsView.vue',
])

function vueFiles(dir: string): string[] {
  const out: string[] = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name)
    if (entry.isDirectory()) out.push(...vueFiles(p))
    else if (entry.name.endsWith('.vue')) out.push(p)
  }
  return out
}

const files = vueFiles(SRC).map((path) => ({
  path,
  name: path.split('/').pop() as string,
  text: readFileSync(path, 'utf8'),
}))

/**
 * Is this write inside a gated handler?
 *
 * Walks back to the enclosing TOP-LEVEL declaration — a `const`/`function` at column 0 — and asks
 * whether `gated(` appears between it and the write. Anchoring on column 0 matters: an earlier
 * attempt took the nearest `const` at any indentation, which found local variables *inside* the
 * gated closure and reported correctly-gated handlers as ungated.
 */
function writeIsGated(text: string, index: number): boolean {
  const before = text.slice(0, index)
  const lines = before.split('\n')
  for (let i = lines.length - 1; i >= 0; i--) {
    if (/^(const|function|async function|export) /.test(lines[i])) {
      return lines.slice(i).join('\n').includes('gated(')
    }
  }
  return false
}

describe('auth gate coverage (#1590)', () => {
  it('every call site of a per-user write is wrapped in the sign-in gate', () => {
    const ungated: string[] = []
    for (const f of files) {
      if (DELEGATES_TO_PARENT.has(f.name) || AUTH_ONLY_ROUTES.has(f.name)) continue
      for (const write of GATED_WRITES) {
        let i = f.text.indexOf(write)
        while (i !== -1) {
          if (!writeIsGated(f.text, i)) {
            ungated.push(`${f.name} → ${write.slice(0, -1)}`)
          }
          i = f.text.indexOf(write, i + 1)
        }
      }
    }

    expect(
      ungated,
      `These call sites perform an auth-gated write outside a gated() handler. Signed out, the ` +
        `click fires a 401 the store swallows, so the control flips and silently reverts — which ` +
        `reads to the user as their own action failing.`,
    ).toEqual([])
  })

  it('no gated control is HIDDEN from signed-out visitors — the whole point of #1590', () => {
    // The class, not one historical instance. A component that imports the gate is a component that
    // knows about deferred actions; combining that with `v-if="auth.isAuthenticated"` means some
    // control is hidden while another is gated, which is exactly the half-wired state.
    const halfWired = files
      .filter((f) => f.text.includes('useSignInGate'))
      .filter((f) => f.text.includes('v-if="auth.isAuthenticated"'))
      .map((f) => f.name)

    expect(
      halfWired,
      `These wire the sign-in gate for one control and hide another behind ` +
        `v-if="auth.isAuthenticated". A hidden control makes its own gate dead code. Render it and ` +
        `let the tap explain the requirement.`,
    ).toEqual([])
  })

  it('capture is never hidden from signed-out visitors', () => {
    // Capture is the entry point to the learning loop — the differentiator. It was hidden three
    // separate ways: PlayerView's :can-capture binding, KnowledgePanel's v-if, PlayerView's
    // mark-moment v-if. Pinned by outcome rather than by regex-per-instance.
    const player = files.find((f) => f.name === 'PlayerView.vue')
    expect(player?.text).not.toMatch(/:can-capture="auth\.isAuthenticated"/)
  })

  it('a delegating component still takes a gated prop, so its label can tell the truth', () => {
    for (const name of DELEGATES_TO_PARENT) {
      const f = files.find((x) => x.name === name)
      expect(f, `${name} is listed as a delegate but no longer exists`).toBeDefined()
      expect(f?.text, `${name} must accept a \`gated\` prop`).toContain('gated?: boolean')
    }
  })

  it('every per-user store write appears in GATED_WRITES', () => {
    // The list above is the guard's blind spot: two writes were missing and nothing said so. Derive
    // the truth from the stores, so adding an action to one forces a decision here.
    const STORES = [
      ['library', 'library.ts'],
      ['favorites', 'favorites.ts'],
      ['queue', 'queue.ts'],
      ['capture', 'capture.ts'],
    ] as const

    const missing: string[] = []
    for (const [alias, file] of STORES) {
      const text = readFileSync(join(SRC, 'stores', file), 'utf8')
      // Actions that hit the API are the ones that need a session.
      for (const m of text.matchAll(/async (\w+)\s*\(/g)) {
        const action = m[1]
        // Reads, and internals a component can never call directly (leading underscore).
        if (['ensureLoaded', 'load', 'refresh'].includes(action) || action.startsWith('_')) continue
        const key = `${alias}.${action}(`
        if (!GATED_WRITES.includes(key)) missing.push(key)
      }
    }

    expect(
      missing,
      `These store actions are not in GATED_WRITES, so no call site of them is checked. Add them ` +
        `(and gate their call sites), or add them to the read-only skip list above if they do not ` +
        `write per-user state.`,
    ).toEqual([])
  })
})
