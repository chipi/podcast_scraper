// @vitest-environment node
import { describe, expect, it } from 'vitest'
import cytoscape from 'cytoscape'
// Importing the options module also registers the fcose extension (#967).
import { giKgCoseLayoutOptionsMain } from './cyCoseLayoutOptions'

/**
 * How long this machine takes to do a fixed piece of CPU-bound work, right now.
 *
 * The perf assertion below is expressed as a multiple of this rather than in milliseconds, so a
 * loaded machine slows the budget and the measurement together instead of turning a healthy layout
 * red. Same process, same thread, same kind of work as the layout itself.
 */
function calibrationMs(): number {
  const t0 = Date.now()
  let acc = 0
  for (let i = 0; i < 4_000_000; i++) acc += Math.sqrt(i % 97)
  // Consume the result so nothing can optimise the loop away.
  if (!Number.isFinite(acc)) throw new Error('calibration did not run')
  return Math.max(1, Date.now() - t0)
}

describe('fcose layout (#967)', () => {
  it('is the configured force-directed layout (not cose)', () => {
    // Guards the cose→fcose swap: if anything reverts the spec name to ``cose`` the
    // O(n²) freeze comes back at scale. The perf test below is the runtime backstop.
    expect(giKgCoseLayoutOptionsMain().name).toBe('fcose')
  })

  it('lays out ~1500 nodes headless within a perf budget', async () => {
    const opts = giKgCoseLayoutOptionsMain()
    const N = 1500
    const elements: cytoscape.ElementDefinition[] = []
    for (let i = 0; i < N; i++) elements.push({ data: { id: `n${i}` } })
    // Connected graph (each node links back to one of 60 hubs) so the layout has real
    // structure to resolve — the regime where cose's all-pairs repulsion explodes.
    for (let i = 1; i < N; i++) elements.push({ data: { source: `n${i}`, target: `n${i % 60}` } })

    const cy = cytoscape({ headless: true, elements })
    // Measured BEFORE the layout so the budget reflects the machine this run is actually getting.
    const budgetMs = calibrationMs() * 660
    try {
      const t0 = Date.now()
      await new Promise<void>((resolve, reject) => {
        // Comfortably past the budget: this guard exists for a layout that never finishes, and it
        // must not fire first and mask the assertion with a less informative error.
        const hardStopMs = Math.max(60_000, budgetMs * 2)
        const timer = setTimeout(
          () => reject(new Error(`fcose layout did not reach layoutstop within ${hardStopMs}ms`)),
          hardStopMs,
        )
        const lo = cy.elements().layout({ ...opts, animate: false } as never)
        lo.one('layoutstop', () => {
          clearTimeout(timer)
          resolve()
        })
        lo.run()
      })
      const dt = Date.now() - t0
      // What this guards: a swap back to `cose`, whose all-pairs repulsion explodes at scale.
      //
      // THE BUDGET IS A MULTIPLE OF THE MACHINE'S OWN SPEED, NOT A WALL CLOCK. A fixed ceiling
      // measures the machine as much as the code: 15s flaked under `make ci-fast`, was raised to
      // 45s, and then failed at 61s on a loaded 14-core box while the same layout took 1.4s in
      // isolation — a red suite for a healthy layout, which trains people to re-run rather than
      // read. Raising the number again would only move the next flake.
      //
      // Measured here (idle), with the calibration loop below as the unit:
      //     calibration        9 ms
      //     fcose  1500 nodes  1.3-1.5 s   ~160x calibration
      //     cose   1500 nodes  15.3 s      ~1700x calibration
      // 660x sits ~4x above a healthy fcose and ~2.6x below the cose regression, and both sides
      // scale with the machine because both are CPU-bound work in this same process.
      expect(dt).toBeLessThan(budgetMs)
      // Layout actually ran (positions assigned, not the 0,0 default for everything).
      const p = cy.getElementById('n0').position()
      expect(Number.isFinite(p.x) && Number.isFinite(p.y)).toBe(true)
    } finally {
      cy.destroy()
    }
  }, 300_000)
})
