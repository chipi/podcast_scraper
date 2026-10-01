import { describe, expect, it } from 'vitest'

/**
 * Every on/off control behaves the same way (operator 2026-09-30: "consistency above all").
 *
 * The follow, heart, follow-show, played, queue and save-insight toggles all flip on the tap and
 * then write through ONE shared serializer (`services/serialWrites`): writes reach the server in
 * the user's order, and a superseded tap's response cannot overwrite a newer one. They used to do
 * this four different ways, and five of the six got a quick tap-tap wrong. `stores/tapTap.test.ts`
 * proves the behaviour per store; this pins the rule itself, so the NEXT toggle added to a store
 * cannot quietly go back to firing requests in parallel.
 */
const stores = import.meta.glob('../stores/*.ts', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

describe('toggle consistency', () => {
  const withToggle = Object.entries(stores).filter(
    ([file, src]) => !file.endsWith('.test.ts') && /async (toggle|captureInsight|captureSpan)\(/.test(src),
  )

  it('finds the toggle stores (not vacuous)', () => {
    const names = withToggle.map(([f]) => f.split('/').pop())
    for (const n of ['interests.ts', 'favorites.ts', 'library.ts', 'completed.ts', 'queue.ts', 'capture.ts']) {
      expect(names, `${n} no longer has a toggle this check recognises`).toContain(n)
    }
  })

  it.each(withToggle.map(([f, src]) => [f.split('/').pop()!, src]))(
    '%s writes through the shared serializer',
    (_name, src) => {
      expect(src).toContain("from '../services/serialWrites'")
      expect(src).toMatch(/writes\.run\(/)
      // And its LOAD goes through `fresh`, so a list fetched before a tap cannot undo the tap.
      expect(src, 'load() must fetch through writes.fresh(...)').toMatch(/writes\.fresh\(/)
    },
  )
})
