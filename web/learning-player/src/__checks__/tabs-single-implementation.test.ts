import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

/**
 * Guardrail (#1594 item 7) — there is ONE tab strip.
 *
 * ## Why a guard and not just the refactor
 *
 * The refactor is the easy half. The app got to SEVEN hand-written tab strips one reasonable commit
 * at a time: each author needed a switcher, the nearest one was a dozen lines of markup, and
 * copying it was faster than finding a shared component. Every copy then missed the same two rules
 * — roving tabindex and the `aria-controls`/`aria-labelledby` pair — because neither is visible
 * unless you are already navigating by keyboard.
 *
 * Nothing about that pressure changed by deleting the copies. Without this check the eighth one
 * lands the same way, and the only person who notices is a keyboard user who will not file a bug.
 *
 * ## What it permits
 *
 * `Tabs.vue` itself, obviously. And TEST files, which legitimately assert on the roles the
 * component emits — a check that forbade `role="tab"` in tests would forbid testing the component.
 */

const SRC = path.join(path.dirname(fileURLToPath(import.meta.url)), '..')

/** The component that is allowed to emit these roles. */
const OWNER = 'components/Tabs.vue'

function walk(dir: string): string[] {
  const out: string[] = []
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name)
    if (e.isDirectory()) out.push(...walk(full))
    else if (/\.(vue|ts)$/.test(e.name) && !/\.test\.ts$/.test(e.name)) out.push(full)
  }
  return out
}

/**
 * Comments are stripped before matching.
 *
 * Three earlier guards in this repo matched their own explanation — a comment saying "never write
 * `role=tablist` by hand" is not an instance of writing it — so the prose that documents a rule
 * kept failing the rule. Cheap to prevent, confusing every time it happens.
 */
function code(src: string): string {
  return src
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '')
}

describe('one tab implementation (#1594 item 7)', () => {
  const files = walk(SRC).filter((f) => !f.endsWith(OWNER))

  it('nothing outside Tabs.vue hand-rolls a tablist or radiogroup', () => {
    const offenders = files.filter((f) => {
      const c = code(readFileSync(f, 'utf8'))
      return /role="(tablist|radiogroup)"/.test(c) || /role="(tab|radio)"/.test(c)
    })
    expect(
      offenders.map((f) => path.relative(SRC, f)),
      'These build a tab strip by hand. Use `Tabs.vue` — it carries the roving tabindex, the ' +
        'arrow keys and the tab↔panel linkage that all seven previous copies were missing. If a ' +
        'genuinely different pattern is needed, extend Tabs rather than starting an eighth.',
    ).toEqual([])
  })

  it('the sweep is actually looking at the app, not an empty list', () => {
    // `offenders === []` is satisfied perfectly by a walk that found no files.
    expect(files.length, 'the file walk found nothing — the check above is vacuous').toBeGreaterThan(50)
  })

  it('Tabs.vue really does emit those roles — so the rule protects something', () => {
    // The mirror risk: if the owner stopped emitting them, the check above would pass forever
    // while the app had no accessible tab strip at all.
    const owner = readFileSync(path.join(SRC, OWNER), 'utf8')
    expect(owner).toContain("'radiogroup' : 'tablist'")
    expect(owner).toContain("'radio' : 'tab'")
  })
})
