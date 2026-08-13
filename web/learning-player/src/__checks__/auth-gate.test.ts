import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Every per-user write is reachable signed-out, and must be gated (#1590).
 *
 * #1590 replaced `v-if="auth.isAuthenticated"` with visible sign-in teasers: the control is not
 * unavailable, it is *deferred*, and hiding it hid the differentiators from exactly the visitors
 * deciding whether to sign up. But that change has a cost — every gated control is now clickable by
 * someone with no session, so **every** call site needs the gate. I wired two of them (queue,
 * favourites) and missed four: both follow buttons and both capture controls. Nothing failed,
 * because nothing asserted it. The stores swallow errors, so the visible result was a control that
 * flipped optimistically, took a 401, and silently flipped back.
 *
 * This is the guard that would have caught me. It is deliberately source-level rather than
 * behavioural: the failure mode is a *missing* call site, and a behavioural test can only cover the
 * call sites someone remembered to write a test for — the same blind spot that caused the bug.
 *
 * Escape hatch: a component that only renders the affordance and emits (the parent owns the API
 * call, so the parent owns the gate) declares a `gated` prop instead. See TranscriptList.
 */

const SRC = join(__dirname, '..')

/**
 * Store actions that write per-user state. Each maps to an authenticated endpoint, so calling one
 * without a session is a 401 — i.e. a bug, never merely a no-op (T7: fix the cause).
 */
const GATED_WRITES = [
  'library.toggle(',
  'favorites.toggle(',
  'favorites.toggleInsight(',
  'queue.add(',
  'queue.remove(',
  'queue.toggle(',
  'capture.captureSpan(',
  'capture.captureInsight(',
]

/** Components that render the affordance but delegate the action (and the gate) to a parent. */
const DELEGATES_TO_PARENT = new Set(['TranscriptList.vue'])

function vueFiles(dir: string): string[] {
  const out: string[] = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name)
    if (entry.isDirectory()) out.push(...vueFiles(p))
    else if (entry.name.endsWith('.vue')) out.push(p)
  }
  return out
}

describe('auth gate coverage (#1590)', () => {
  const files = vueFiles(SRC).map((path) => ({
    path,
    name: path.split('/').pop() as string,
    text: readFileSync(path, 'utf8'),
  }))

  it('every component that performs a per-user write routes it through the sign-in gate', () => {
    const ungated = files
      .filter((f) => GATED_WRITES.some((w) => f.text.includes(w)))
      .filter((f) => !f.text.includes('useSignInGate'))
      .filter((f) => !DELEGATES_TO_PARENT.has(f.name))
      .map((f) => f.name)

    expect(
      ungated,
      `These perform an auth-gated write without useSignInGate. Signed out, the click fires a 401 ` +
        `the store swallows, so the control flips and silently reverts. Wire the gate, or — if the ` +
        `component only emits — add it to DELEGATES_TO_PARENT and take a \`gated\` prop.`,
    ).toEqual([])
  })

  it('a delegating component still takes a gated prop, so its label can tell the truth', () => {
    for (const name of DELEGATES_TO_PARENT) {
      const f = files.find((x) => x.name === name)
      expect(f, `${name} is listed as a delegate but no longer exists`).toBeDefined()
      expect(f?.text, `${name} must accept a \`gated\` prop`).toContain('gated?: boolean')
    }
  })

  it('no gated control is hidden from signed-out visitors — the point of #1590', () => {
    // The regression #1590 exists to prevent: re-hiding a differentiator behind isAuthenticated.
    // PlayerView passed `:can-capture="auth.isAuthenticated"`, so the capture affordance — the entry
    // point to the learning loop — did not exist for signed-out readers of the transcript.
    const player = files.find((f) => f.name === 'PlayerView.vue')
    expect(player?.text).not.toMatch(/:can-capture="auth\.isAuthenticated"/)
  })
})
