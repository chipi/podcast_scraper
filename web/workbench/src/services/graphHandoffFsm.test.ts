/**
 * Unit tests for the pure handoff FSM (`services/graphHandoffFsm.ts`).
 *
 * Coverage contract: every (state × event) cell that is *valid* per the spec
 * has a positive test; every *invalid* combination has a negative test.
 * Transition table: 8 states × 9 events ≈ 72 cells, ~42 valid, ~30 invalid.
 *
 * Plus tests for envelope validation, generation supersession, re-entrance
 * policies (decision #5 / FSM spec), and `isValidAdvance`.
 */

import { describe, expect, it } from 'vitest'
import {
  advanceState,
  applyEvent,
  createFsm,
  isStale,
  isValidAdvance,
  validateEnvelope,
  type EnvelopeInput,
  type FsmState,
} from './graphHandoffFsm'

function envelope(over: Partial<EnvelopeInput> = {}): EnvelopeInput {
  return {
    kind: 'episode',
    cyId: 'g:episode:abc',
    metadataPath: 'metadata/abc.metadata.json',
    episodeId: 'abc',
    source: 'library',
    loadSource: 'subject-external',
    camera: { kind: 'center', cyId: 'g:episode:abc' },
    ...over,
  }
}

// ---------------------------------------------------------------------------
// Envelope validation
// ---------------------------------------------------------------------------

describe('validateEnvelope', () => {
  it('episode envelope accepts cyId only', () => {
    expect(
      validateEnvelope(
        envelope({ metadataPath: undefined, episodeId: undefined }),
      ),
    ).toBeNull()
  })

  it('episode envelope accepts metadataPath only', () => {
    expect(
      validateEnvelope(envelope({ cyId: undefined, episodeId: undefined })),
    ).toBeNull()
  })

  it('episode envelope accepts episodeId only', () => {
    expect(
      validateEnvelope(envelope({ cyId: undefined, metadataPath: undefined })),
    ).toBeNull()
  })

  it('episode envelope rejects all-empty', () => {
    expect(
      validateEnvelope(
        envelope({ cyId: '', metadataPath: '', episodeId: '' }),
      ),
    ).toMatch(/episode envelope requires/)
  })

  it('graph-node envelope requires cyId', () => {
    expect(
      validateEnvelope(envelope({ kind: 'graph-node', cyId: '' })),
    ).toMatch(/graph-node envelope requires cyId/)
  })

  it('graph-node envelope accepts cyId', () => {
    expect(
      validateEnvelope(envelope({ kind: 'graph-node', cyId: 'g:topic:t1' })),
    ).toBeNull()
  })

  it('topic envelope requires cyId', () => {
    expect(
      validateEnvelope(envelope({ kind: 'topic', cyId: '' })),
    ).toMatch(/topic envelope requires cyId/)
  })

  it('camera center requires non-empty cyId', () => {
    expect(
      validateEnvelope(
        envelope({ camera: { kind: 'center', cyId: '   ' } }),
      ),
    ).toMatch(/camera strategy 'center' requires cyId/)
  })

  it('camera fit / preserve / none accept any envelope', () => {
    expect(
      validateEnvelope(envelope({ camera: { kind: 'fit' } })),
    ).toBeNull()
    expect(
      validateEnvelope(envelope({ camera: { kind: 'preserve' } })),
    ).toBeNull()
    expect(validateEnvelope(envelope({ camera: { kind: 'none' } }))).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// handoffRequested
// ---------------------------------------------------------------------------

describe('handoffRequested', () => {
  it('from idle: accepts and stamps generation 1', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, { type: 'handoffRequested', envelope: envelope() })
    expect(disp.kind).toBe('accept')
    if (disp.kind !== 'accept') return
    expect(disp.envelope?.generation).toBe(1)
    expect(disp.supersededInFlight).toBe(false)
    expect(fsm.state).toBe('loading_fetch')
  })

  it('from ready: accepts and bumps generation', () => {
    const fsm = createFsm()
    fsm.state = 'ready'
    const disp = applyEvent(fsm, { type: 'handoffRequested', envelope: envelope() })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('loading_fetch')
    expect(fsm.generation).toBe(1)
  })

  it('mid-load: supersedes in-flight', () => {
    const fsm = createFsm()
    applyEvent(fsm, { type: 'handoffRequested', envelope: envelope({ cyId: 'first' }) })
    expect(fsm.state).toBe('loading_fetch')
    const disp = applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ cyId: 'second' }),
    })
    expect(disp.kind).toBe('accept')
    if (disp.kind !== 'accept') return
    expect(disp.supersededInFlight).toBe(true)
    expect(disp.envelope?.cyId).toBe('second')
    expect(fsm.generation).toBe(2)
  })

  it('canvas-tap source: jumps to applying', () => {
    const fsm = createFsm()
    fsm.state = 'ready'
    const disp = applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ source: 'canvas-tap', kind: 'graph-node' }),
    })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('applying')
  })

  it('minimap source: jumps to applying', () => {
    const fsm = createFsm()
    fsm.state = 'ready'
    applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ source: 'minimap', kind: 'graph-node' }),
    })
    expect(fsm.state).toBe('applying')
  })

  it('rejects invalid envelope (returns fail)', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ kind: 'graph-node', cyId: '' }),
    })
    expect(disp.kind).toBe('fail')
  })
})

// ---------------------------------------------------------------------------
// canvasTapped (re-entrance: queue same-target / supersede different-target)
// ---------------------------------------------------------------------------

describe('canvasTapped', () => {
  it('from ready: jumps to applying', () => {
    const fsm = createFsm()
    fsm.state = 'ready'
    const disp = applyEvent(fsm, {
      type: 'canvasTapped',
      envelope: envelope({ source: 'canvas-tap', kind: 'graph-node' }),
    })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('applying')
  })

  it('mid-load same target: queues', () => {
    const fsm = createFsm()
    applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ cyId: 'shared-id' }),
    })
    const disp = applyEvent(fsm, {
      type: 'canvasTapped',
      envelope: envelope({
        source: 'canvas-tap',
        kind: 'episode',
        cyId: 'shared-id',
      }),
    })
    expect(disp.kind).toBe('queue')
  })

  it('mid-load different target: supersedes', () => {
    const fsm = createFsm()
    applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ cyId: 'A' }),
    })
    const beforeGen = fsm.generation
    const disp = applyEvent(fsm, {
      type: 'canvasTapped',
      envelope: envelope({
        source: 'canvas-tap',
        kind: 'graph-node',
        cyId: 'B',
      }),
    })
    expect(disp.kind).toBe('accept')
    if (disp.kind !== 'accept') return
    expect(disp.supersededInFlight).toBe(true)
    expect(fsm.generation).toBeGreaterThan(beforeGen)
    expect(fsm.state).toBe('applying')
  })

  it('mid-load different kind: supersedes', () => {
    const fsm = createFsm()
    applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ kind: 'episode', cyId: 'shared' }),
    })
    const disp = applyEvent(fsm, {
      type: 'canvasTapped',
      envelope: envelope({ kind: 'graph-node', cyId: 'shared' }),
    })
    expect(disp.kind).toBe('accept')
  })

  it('rejects invalid envelope', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'canvasTapped',
      envelope: envelope({ kind: 'graph-node', cyId: '' }),
    })
    expect(disp.kind).toBe('fail')
  })
})

// ---------------------------------------------------------------------------
// expansionRequested (always queue if in flight; additive)
// ---------------------------------------------------------------------------

describe('expansionRequested', () => {
  it('from ready: enters loading_fetch', () => {
    const fsm = createFsm()
    fsm.state = 'ready'
    const disp = applyEvent(fsm, {
      type: 'expansionRequested',
      envelope: envelope({ source: 'double-tap-expand', kind: 'graph-node' }),
    })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('loading_fetch')
  })

  it('from idle: enters loading_fetch', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'expansionRequested',
      envelope: envelope({ source: 'double-tap-expand', kind: 'graph-node' }),
    })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('loading_fetch')
  })

  it('mid-load: queues regardless of target', () => {
    const fsm = createFsm()
    applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ cyId: 'A' }),
    })
    const disp = applyEvent(fsm, {
      type: 'expansionRequested',
      envelope: envelope({
        source: 'double-tap-expand',
        kind: 'graph-node',
        cyId: 'B',
      }),
    })
    expect(disp.kind).toBe('queue')
  })

  it('rejects invalid envelope', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'expansionRequested',
      envelope: envelope({ kind: 'topic', cyId: '' }),
    })
    expect(disp.kind).toBe('fail')
  })
})

// ---------------------------------------------------------------------------
// focusCleared (always supersede with empty envelope; → ready)
// ---------------------------------------------------------------------------

describe('focusCleared', () => {
  it('from ready: stays ready, bumps generation', () => {
    const fsm = createFsm()
    fsm.state = 'ready'
    const disp = applyEvent(fsm, { type: 'focusCleared' })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('ready')
    expect(fsm.pending).toBeNull()
  })

  it('mid-load: supersedes and clears', () => {
    const fsm = createFsm()
    applyEvent(fsm, { type: 'handoffRequested', envelope: envelope() })
    const disp = applyEvent(fsm, { type: 'focusCleared' })
    expect(disp.kind).toBe('accept')
    if (disp.kind !== 'accept') return
    expect(disp.supersededInFlight).toBe(true)
    expect(fsm.state).toBe('ready')
    expect(fsm.pending).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// tabReturned (drop if not ready)
// ---------------------------------------------------------------------------

describe('tabReturned', () => {
  it('from ready: accepted (reconcile-only)', () => {
    const fsm = createFsm()
    fsm.state = 'ready'
    const disp = applyEvent(fsm, { type: 'tabReturned' })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('ready')
  })

  it('from idle: dropped', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, { type: 'tabReturned' })
    expect(disp.kind).toBe('drop')
  })

  it('from loading_fetch: dropped', () => {
    const fsm = createFsm()
    fsm.state = 'loading_fetch'
    const disp = applyEvent(fsm, { type: 'tabReturned' })
    expect(disp.kind).toBe('drop')
  })

  it('from applying: dropped', () => {
    const fsm = createFsm()
    fsm.state = 'applying'
    const disp = applyEvent(fsm, { type: 'tabReturned' })
    expect(disp.kind).toBe('drop')
  })
})

// ---------------------------------------------------------------------------
// corpusReloaded (full reset)
// ---------------------------------------------------------------------------

describe('corpusReloaded', () => {
  it('from any state resets to idle and bumps generation', () => {
    const fsm = createFsm()
    fsm.state = 'applying'
    fsm.pending = { ...envelope(), generation: 5 }
    fsm.generation = 5
    const disp = applyEvent(fsm, { type: 'corpusReloaded' })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('idle')
    expect(fsm.pending).toBeNull()
    expect(fsm.generation).toBe(6)
  })
})

// ---------------------------------------------------------------------------
// handoffFailed (preserve previous selection; → ready)
// ---------------------------------------------------------------------------

describe('handoffFailed', () => {
  it('clears pending and returns to ready', () => {
    const fsm = createFsm()
    applyEvent(fsm, { type: 'handoffRequested', envelope: envelope() })
    const disp = applyEvent(fsm, { type: 'handoffFailed', reason: 'mock 404' })
    expect(disp.kind).toBe('fail')
    expect(fsm.state).toBe('ready')
    expect(fsm.pending).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// layoutstop / layoutstart
// ---------------------------------------------------------------------------

describe('layoutstop', () => {
  it('from redrawing_incremental: → applying', () => {
    const fsm = createFsm()
    fsm.state = 'redrawing_incremental'
    const disp = applyEvent(fsm, { type: 'layoutstop' })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('applying')
  })

  it('from redrawing_full: → applying', () => {
    const fsm = createFsm()
    fsm.state = 'redrawing_full'
    const disp = applyEvent(fsm, { type: 'layoutstop' })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('applying')
  })

  it('from applying: → ready', () => {
    const fsm = createFsm()
    fsm.state = 'applying'
    const disp = applyEvent(fsm, { type: 'layoutstop' })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('ready')
  })

  it('from idle: dropped', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, { type: 'layoutstop' })
    expect(disp.kind).toBe('drop')
  })

  it('from ready: dropped', () => {
    const fsm = createFsm()
    fsm.state = 'ready'
    const disp = applyEvent(fsm, { type: 'layoutstop' })
    expect(disp.kind).toBe('drop')
  })

  it('from loading_*: dropped', () => {
    for (const s of ['loading_fetch', 'loading_bootstrap', 'loading_merge'] as const) {
      const fsm = createFsm()
      fsm.state = s
      const disp = applyEvent(fsm, { type: 'layoutstop' })
      expect(disp.kind).toBe('drop')
    }
  })
})

describe('layoutstart', () => {
  it('from applying: → redrawing_full', () => {
    const fsm = createFsm()
    fsm.state = 'applying'
    const disp = applyEvent(fsm, { type: 'layoutstart' })
    expect(disp.kind).toBe('accept')
    expect(fsm.state).toBe('redrawing_full')
  })

  it('from any other state: dropped', () => {
    for (const s of [
      'idle',
      'ready',
      'loading_fetch',
      'loading_bootstrap',
      'loading_merge',
      'redrawing_incremental',
      'redrawing_full',
    ] as const) {
      const fsm = createFsm()
      fsm.state = s
      const disp = applyEvent(fsm, { type: 'layoutstart' })
      expect(disp.kind).toBe('drop')
    }
  })
})

// ---------------------------------------------------------------------------
// Generation supersession (isStale)
// ---------------------------------------------------------------------------

describe('isStale', () => {
  it('matches current generation: not stale', () => {
    const fsm = createFsm()
    applyEvent(fsm, { type: 'handoffRequested', envelope: envelope() })
    expect(isStale(fsm, fsm.generation)).toBe(false)
  })

  it('older generation: stale', () => {
    const fsm = createFsm()
    applyEvent(fsm, { type: 'handoffRequested', envelope: envelope() })
    const oldGen = fsm.generation
    applyEvent(fsm, { type: 'handoffRequested', envelope: envelope({ cyId: 'next' }) })
    expect(isStale(fsm, oldGen)).toBe(true)
    expect(isStale(fsm, fsm.generation)).toBe(false)
  })

  it('5 rapid handoffs: only last is fresh', () => {
    const fsm = createFsm()
    const gens: number[] = []
    for (let i = 0; i < 5; i++) {
      const disp = applyEvent(fsm, {
        type: 'handoffRequested',
        envelope: envelope({ cyId: `node-${i}` }),
      })
      if (disp.kind === 'accept' && disp.envelope) {
        gens.push(disp.envelope.generation)
      }
    }
    expect(gens.length).toBe(5)
    for (let i = 0; i < 4; i++) {
      expect(isStale(fsm, gens[i]!)).toBe(true)
    }
    expect(isStale(fsm, gens[4]!)).toBe(false)
  })

  it('focusCleared: bumps generation, in-flight envelope becomes stale', () => {
    const fsm = createFsm()
    applyEvent(fsm, { type: 'handoffRequested', envelope: envelope() })
    const oldGen = fsm.generation
    applyEvent(fsm, { type: 'focusCleared' })
    expect(isStale(fsm, oldGen)).toBe(true)
  })

  it('corpusReloaded: bumps generation, all in-flight stale', () => {
    const fsm = createFsm()
    applyEvent(fsm, { type: 'handoffRequested', envelope: envelope() })
    const oldGen = fsm.generation
    applyEvent(fsm, { type: 'corpusReloaded' })
    expect(isStale(fsm, oldGen)).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// advanceState (forward transitions during load/redraw/apply)
// ---------------------------------------------------------------------------

describe('advanceState / isValidAdvance', () => {
  it('idle → loading_fetch valid', () => {
    expect(isValidAdvance('idle', 'loading_fetch')).toBe(true)
  })

  it('loading_fetch → loading_bootstrap valid', () => {
    expect(isValidAdvance('loading_fetch', 'loading_bootstrap')).toBe(true)
  })

  it('loading_fetch → loading_merge valid (skip bootstrap)', () => {
    expect(isValidAdvance('loading_fetch', 'loading_merge')).toBe(true)
  })

  it('loading_bootstrap → loading_merge valid', () => {
    expect(isValidAdvance('loading_bootstrap', 'loading_merge')).toBe(true)
  })

  it('loading_merge → redrawing_incremental valid', () => {
    expect(isValidAdvance('loading_merge', 'redrawing_incremental')).toBe(true)
  })

  it('loading_merge → redrawing_full valid', () => {
    expect(isValidAdvance('loading_merge', 'redrawing_full')).toBe(true)
  })

  it('redrawing_incremental → applying valid', () => {
    expect(isValidAdvance('redrawing_incremental', 'applying')).toBe(true)
  })

  it('redrawing_full → applying valid', () => {
    expect(isValidAdvance('redrawing_full', 'applying')).toBe(true)
  })

  it('applying → ready valid', () => {
    expect(isValidAdvance('applying', 'ready')).toBe(true)
  })

  it('ready → loading_fetch valid (next handoff)', () => {
    expect(isValidAdvance('ready', 'loading_fetch')).toBe(true)
  })

  it('idle → applying valid (canvas tap path)', () => {
    expect(isValidAdvance('idle', 'applying')).toBe(true)
  })

  it('ready → applying valid (canvas tap from quiescent)', () => {
    expect(isValidAdvance('ready', 'applying')).toBe(true)
  })

  it('rejects invalid: redrawing_incremental → idle', () => {
    expect(isValidAdvance('redrawing_incremental', 'idle')).toBe(false)
  })

  it('rejects invalid: applying → loading_fetch', () => {
    expect(isValidAdvance('applying', 'loading_fetch')).toBe(false)
  })

  it('rejects invalid: ready → applying.selection (no such state)', () => {
    expect(isValidAdvance('ready', 'applying.selection' as FsmState)).toBe(false)
  })

  it('advanceState mutates and returns target on valid', () => {
    const fsm = createFsm()
    fsm.state = 'loading_fetch'
    expect(advanceState(fsm, 'loading_merge')).toBe('loading_merge')
    expect(fsm.state).toBe('loading_merge')
  })

  it('advanceState returns null on invalid', () => {
    const fsm = createFsm()
    fsm.state = 'idle'
    expect(advanceState(fsm, 'redrawing_incremental')).toBeNull()
    expect(fsm.state).toBe('idle')
  })
})

// ---------------------------------------------------------------------------
// Re-entrance policy (per decision #5 / FSM spec)
// ---------------------------------------------------------------------------

describe('re-entrance policy', () => {
  it('handoffRequested always supersedes (3 in a row)', () => {
    const fsm = createFsm()
    applyEvent(fsm, { type: 'handoffRequested', envelope: envelope({ cyId: 'A' }) })
    applyEvent(fsm, { type: 'handoffRequested', envelope: envelope({ cyId: 'B' }) })
    const disp = applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ cyId: 'C' }),
    })
    expect(disp.kind).toBe('accept')
    if (disp.kind !== 'accept') return
    expect(disp.envelope?.cyId).toBe('C')
    expect(fsm.generation).toBe(3)
  })

  it('canvasTapped queue same-target idempotency', () => {
    const fsm = createFsm()
    applyEvent(fsm, {
      type: 'canvasTapped',
      envelope: envelope({ source: 'canvas-tap', kind: 'graph-node', cyId: 'X' }),
    })
    // FSM is now in applying. A second canvasTapped on the same X queues.
    const disp = applyEvent(fsm, {
      type: 'canvasTapped',
      envelope: envelope({ source: 'canvas-tap', kind: 'graph-node', cyId: 'X' }),
    })
    expect(disp.kind).toBe('queue')
  })

  it('expansionRequested always queues (3 in flight)', () => {
    const fsm = createFsm()
    applyEvent(fsm, {
      type: 'expansionRequested',
      envelope: envelope({ source: 'double-tap-expand', kind: 'graph-node', cyId: 'A' }),
    })
    const second = applyEvent(fsm, {
      type: 'expansionRequested',
      envelope: envelope({ source: 'double-tap-expand', kind: 'graph-node', cyId: 'B' }),
    })
    const third = applyEvent(fsm, {
      type: 'expansionRequested',
      envelope: envelope({ source: 'double-tap-expand', kind: 'graph-node', cyId: 'C' }),
    })
    expect(second.kind).toBe('queue')
    expect(third.kind).toBe('queue')
  })
})

// ---------------------------------------------------------------------------
// Camera strategy preservation through stamping
// ---------------------------------------------------------------------------

describe('camera strategy on envelope', () => {
  it('center is preserved through stamping', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({
        camera: { kind: 'center', cyId: 'g:episode:abc', includes: ['x', 'y'] },
      }),
    })
    if (disp.kind !== 'accept' || !disp.envelope) {
      throw new Error('expected accept')
    }
    expect(disp.envelope.camera.kind).toBe('center')
    if (disp.envelope.camera.kind !== 'center') return
    expect(disp.envelope.camera.cyId).toBe('g:episode:abc')
    expect(disp.envelope.camera.includes).toEqual(['x', 'y'])
  })

  it('fit is preserved', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ camera: { kind: 'fit' } }),
    })
    if (disp.kind !== 'accept' || !disp.envelope) {
      throw new Error('expected accept')
    }
    expect(disp.envelope.camera.kind).toBe('fit')
  })

  it('preserve is preserved', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'canvasTapped',
      envelope: envelope({
        source: 'minimap',
        kind: 'graph-node',
        camera: { kind: 'preserve' },
      }),
    })
    if (disp.kind !== 'accept' || !disp.envelope) {
      throw new Error('expected accept')
    }
    expect(disp.envelope.camera.kind).toBe('preserve')
  })
})

// ---------------------------------------------------------------------------
// Highlights through stamping (decision #10)
// ---------------------------------------------------------------------------

describe('highlights on envelope', () => {
  it('highlights preserved through stamping', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ highlights: ['ep1', 'ep2'] }),
    })
    if (disp.kind !== 'accept' || !disp.envelope) {
      throw new Error('expected accept')
    }
    expect(disp.envelope.highlights).toEqual(['ep1', 'ep2'])
  })

  it('absent highlights stay undefined (consumers treat as clear)', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope(),
    })
    if (disp.kind !== 'accept' || !disp.envelope) {
      throw new Error('expected accept')
    }
    expect(disp.envelope.highlights).toBeUndefined()
  })
})

// ---------------------------------------------------------------------------
// suppressCamera (decision #6 — minimap)
// ---------------------------------------------------------------------------

describe('suppressCamera', () => {
  it('preserves on canvasTapped from minimap', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'canvasTapped',
      envelope: envelope({
        source: 'minimap',
        kind: 'graph-node',
        suppressCamera: true,
      }),
    })
    if (disp.kind !== 'accept' || !disp.envelope) {
      throw new Error('expected accept')
    }
    expect(disp.envelope.suppressCamera).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// Source-specific behaviour (decision #2 / load-source granularity)
// ---------------------------------------------------------------------------

describe('source-specific behaviour', () => {
  it('library source: subject-external load source via envelope', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ source: 'library', loadSource: 'subject-external' }),
    })
    if (disp.kind !== 'accept' || !disp.envelope) {
      throw new Error('expected accept')
    }
    expect(disp.envelope.loadSource).toBe('subject-external')
  })

  it('digest source: digest-external', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({ source: 'digest', loadSource: 'digest-external' }),
    })
    if (disp.kind !== 'accept' || !disp.envelope) {
      throw new Error('expected accept')
    }
    expect(disp.envelope.loadSource).toBe('digest-external')
  })

  it('node-detail source via expansionRequested: graph-internal', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'expansionRequested',
      envelope: envelope({
        source: 'node-detail',
        kind: 'graph-node',
        loadSource: 'graph-internal',
      }),
    })
    if (disp.kind !== 'accept' || !disp.envelope) {
      throw new Error('expected accept')
    }
    expect(disp.envelope.loadSource).toBe('graph-internal')
  })

  it('restore-preference source: handled as handoffRequested', () => {
    const fsm = createFsm()
    const disp = applyEvent(fsm, {
      type: 'handoffRequested',
      envelope: envelope({
        source: 'restore-preference',
        loadSource: 'subject-external',
      }),
    })
    expect(disp.kind).toBe('accept')
  })
})
