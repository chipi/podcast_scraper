// @vitest-environment node
import { describe, expect, it } from 'vitest'
import cytoscape from 'cytoscape'
// Importing the options module also registers the fcose extension (#967).
import { giKgCoseLayoutOptionsMain } from './cyCoseLayoutOptions'

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
    try {
      const t0 = Date.now()
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error('fcose layout did not reach layoutstop within 60s')),
          60_000,
        )
        const lo = cy.elements().layout({ ...opts, animate: false } as never)
        lo.one('layoutstop', () => {
          clearTimeout(timer)
          resolve()
        })
        lo.run()
      })
      const dt = Date.now() - t0
      // What this guards: a swap back to `cose`, which took ~134s at ~2.9k nodes where fcose
      // (spectral seed) settles ~1.5k in low single-digit seconds. That is a ~60x gap, so the
      // ceiling only has to land somewhere between — it does not need to be tight.
      //
      // It was 15s, calibrated on an idle machine, and that made it flake: a wall-clock assertion
      // is a measurement of the MACHINE as much as the code. Observed here at load ~29 on 12 cores
      // (the rest of `make ci-fast` running alongside) it exceeded 15s while taking 1.97s in
      // isolation — a red suite for a healthy layout, which is worse than no assertion at all
      // because it trains people to re-run rather than read.
      //
      // 45s keeps ~3x margin below the cose regression it exists to catch and ~20x headroom above
      // a loaded-machine fcose run. Tightening this buys nothing: nothing lives between 2s and
      // 134s to discriminate.
      expect(dt).toBeLessThan(45_000)
      // Layout actually ran (positions assigned, not the 0,0 default for everything).
      const p = cy.getElementById('n0').position()
      expect(Number.isFinite(p.x) && Number.isFinite(p.y)).toBe(true)
    } finally {
      cy.destroy()
    }
  }, 90_000)
})
