import { readFileSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Every store the identity watcher resets must actually BE resettable (#1955).
 *
 * Pinia auto-generates `$reset()` for OPTIONS stores only. A store written with SETUP syntax
 * (`defineStore('x', () => {...})`) throws on `$reset()` unless it implements one itself.
 *
 * That is not a lint nit. `App.vue`'s identity watcher resets each per-user store in sequence,
 * with no try/catch. When `userPreferences` — the one setup store in the list — threw, Vue
 * swallowed it as an unhandled watcher error and ABANDONED the rest of the callback. The lines
 * after it never ran, so `useCaptureStore().$reset()` and `player.clear()` were skipped and A's
 * highlights, private notes and playback survived into B's session.
 *
 * This is a STATIC check on purpose. Instantiating the stores would need a Pinia app, an auth
 * store and fetch mocks, and would test the harness as much as the invariant. Reading the source
 * asserts exactly the thing that broke: this store is reset by the watcher, and this store is a
 * setup store, therefore this store must define `$reset`.
 *
 * It derives the list from App.vue rather than hardcoding it, so adding a store to the watcher
 * automatically brings it under the guard.
 */
const SRC = resolve(__dirname, '..')
const appVue = readFileSync(resolve(SRC, 'App.vue'), 'utf-8')

/** `useLibraryStore().$reset()` / `queue.$reset()` -> the names reset by the watcher. */
function storesResetByTheWatcher(): string[] {
  const viaFactory = [...appVue.matchAll(/use(\w+?)Store\(\)\.\$reset\(\)/g)].map((m) => m[1])
  const viaLocal = [...appVue.matchAll(/^\s*(\w+)\.\$reset\(\)/gm)].map((m) => m[1])
  const lower = (s: string) => s.charAt(0).toLowerCase() + s.slice(1)
  return [...new Set([...viaFactory.map(lower), ...viaLocal])]
}

/**
 * Strip comments before pattern-matching source.
 *
 * Without this the guard is satisfied by prose: a doc comment that merely MENTIONS `$reset()`
 * counts as an implementation. Verified the hard way — the first version passed with the fix
 * deliberately disabled, because this file's own explanatory comment contains the token.
 */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '')
}

/** Setup stores are `defineStore('name', () => ...)`; options stores pass an object. */
function isSetupStore(source: string): boolean {
  return /defineStore\(\s*['"][^'"]+['"]\s*,\s*(async\s*)?\(\s*\)\s*=>/.test(source)
}

function storeFileFor(name: string): string | null {
  const files = readdirSync(resolve(SRC, 'stores'))
  const exact = files.find((f) => f.toLowerCase() === `${name.toLowerCase()}.ts`)
  return exact ? resolve(SRC, 'stores', exact) : null
}

describe('identity reset (#1955)', () => {
  it('finds the stores the identity watcher resets', () => {
    const names = storesResetByTheWatcher()
    // If this drops to zero the regex has drifted from App.vue and the guard below would pass
    // vacuously — which is how a guard quietly stops guarding.
    expect(names.length, 'App.vue should reset several per-user stores').toBeGreaterThan(3)
  })

  it('every setup store reset by the watcher implements $reset', () => {
    const offenders: string[] = []
    for (const name of storesResetByTheWatcher()) {
      const file = storeFileFor(name)
      if (!file) continue // not a store module (e.g. a local player instance)
      const source = stripComments(readFileSync(file, 'utf-8'))
      if (!isSetupStore(source)) continue // options store: Pinia generates $reset
      // Declaration or exported property only. Matching the bare token would be satisfied by a
      // COMMENT mentioning $reset — which is exactly how the first version of this guard passed
      // with the fix deliberately removed.
      if (!/function\s+\$reset\s*\(|(^|[{,\s])\$reset\s*[,:]/m.test(source)) {
        offenders.push(
          `${name}: setup-syntax store reset by App.vue's identity watcher, but defines no ` +
            `$reset() — Pinia will THROW and abandon the rest of the watcher, skipping every ` +
            `store after it (this is #1955)`,
        )
      }
    }
    expect(offenders, offenders.join('\n  ')).toEqual([])
  })
})
