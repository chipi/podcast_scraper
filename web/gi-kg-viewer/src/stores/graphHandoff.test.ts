// @vitest-environment happy-dom
/**
 * T5 — Telemetry firing unit test for the graph handoff store.
 *
 * Asserts each FSM event fires the matching PostHog `graph_handoff_*` event
 * with the expected payload. If a future PR drops a `posthog.capture(...)`
 * call from the store, that event's test goes red — telemetry breakage is
 * caught at PR time rather than weeks later when dashboards go quiet.
 *
 * Pattern: `vi.mock('posthog-js')` to spy on the default export's `capture`
 * method, then drive the store via its public API.
 */

import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({
  default: {
    capture: vi.fn(),
  },
}))

// posthog-js needs to be imported AFTER the vi.mock declaration (Vitest
// hoists `vi.mock` to the top of the module so this works either way; the
// import below resolves to the mocked module).
import posthog from 'posthog-js'
import type { EnvelopeInput } from '../services/graphHandoffFsm'
import { useGraphHandoffStore } from './graphHandoff'

const captureSpy = vi.mocked(posthog.capture)

function envelope(over: Partial<EnvelopeInput> = {}): EnvelopeInput {
  return {
    kind: 'episode',
    cyId: 'g:episode:abc',
    source: 'library',
    loadSource: 'subject-external',
    camera: { kind: 'center-on-target' },
    ...over,
  }
}

describe('useGraphHandoffStore — telemetry (T5)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('handoffRequested fires graph_handoff_started with envelope payload', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope({ source: 'library' }))

    expect(captureSpy).toHaveBeenCalledWith(
      'graph_handoff_started',
      expect.objectContaining({
        source: 'library',
        kind: 'episode',
        load_source: 'subject-external',
        camera_kind: 'center-on-target',
        event_type: 'handoffRequested',
      }),
    )
    const call = captureSpy.mock.calls.find(
      (c) => c[0] === 'graph_handoff_started',
    )
    expect(call).toBeDefined()
    expect(call?.[1]).toHaveProperty('generation')
    expect(typeof (call?.[1] as { generation?: number })?.generation).toBe('number')
  })

  it('canvasTapped fires graph_handoff_started with source=canvas-tap', () => {
    const store = useGraphHandoffStore()
    store.canvasTapped(
      envelope({
        kind: 'graph-node',
        source: 'canvas-tap',
        loadSource: 'graph-internal',
        camera: { kind: 'center', cyId: 'g:topic:x' },
      }),
    )

    expect(captureSpy).toHaveBeenCalledWith(
      'graph_handoff_started',
      expect.objectContaining({
        source: 'canvas-tap',
        kind: 'graph-node',
        load_source: 'graph-internal',
        camera_kind: 'center',
        event_type: 'canvasTapped',
      }),
    )
  })

  it('expansionRequested fires graph_handoff_started with source=double-tap-expand', () => {
    const store = useGraphHandoffStore()
    store.expansionRequested(
      envelope({
        kind: 'graph-node',
        source: 'double-tap-expand',
        loadSource: 'graph-internal',
        camera: { kind: 'preserve' },
      }),
    )

    expect(captureSpy).toHaveBeenCalledWith(
      'graph_handoff_started',
      expect.objectContaining({
        source: 'double-tap-expand',
        kind: 'graph-node',
        load_source: 'graph-internal',
        camera_kind: 'preserve',
      }),
    )
  })

  it('handoffFailed fires graph_handoff_failed with reason', () => {
    const store = useGraphHandoffStore()
    store.handoffFailed('mock 404 from territory fetch')

    expect(captureSpy).toHaveBeenCalledWith(
      'graph_handoff_failed',
      expect.objectContaining({
        reason: 'mock 404 from territory fetch',
        event_type: 'handoffFailed',
      }),
    )
  })

  it('recordApplied fires graph_handoff_applied with applied_cy_id', () => {
    const store = useGraphHandoffStore()
    // Need a pending envelope first so recordApplied has something to record.
    store.handoffRequested(envelope())
    captureSpy.mockClear()
    store.recordApplied('g:episode:abc')

    expect(captureSpy).toHaveBeenCalledWith(
      'graph_handoff_applied',
      expect.objectContaining({
        applied_cy_id: 'g:episode:abc',
        fallback_applied: false,
      }),
    )
  })

  it('handoffRequested while another is in flight reports superseded_in_flight=true', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope({ cyId: 'first' }))
    captureSpy.mockClear()
    store.handoffRequested(envelope({ cyId: 'second' }))

    expect(captureSpy).toHaveBeenCalledWith(
      'graph_handoff_started',
      expect.objectContaining({
        superseded_in_flight: true,
      }),
    )
  })

  it('focusCleared does not emit a started/failed event (event without envelope)', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope())
    captureSpy.mockClear()
    store.focusCleared()

    // focusCleared accepts and clears state but doesn't carry an envelope —
    // store correctly skips the started/failed emit.
    const callsAfterClear = captureSpy.mock.calls.filter(
      (c) => c[0] === 'graph_handoff_started' || c[0] === 'graph_handoff_failed',
    )
    expect(callsAfterClear.length).toBe(0)
  })

  it('stuck-timer firing emits graph_handoff_stuck after STUCK_TIMEOUT_MS', () => {
    vi.useFakeTimers()
    try {
      const store = useGraphHandoffStore()
      store.handoffRequested(envelope({ source: 'library' }))
      captureSpy.mockClear()
      // Advance past the 15s stuck-timeout.
      vi.advanceTimersByTime(15_001)

      expect(captureSpy).toHaveBeenCalledWith(
        'graph_handoff_stuck',
        expect.objectContaining({
          source: 'library',
          kind: 'episode',
          load_source: 'subject-external',
          timeout_ms: 15_000,
        }),
      )
    } finally {
      vi.useRealTimers()
    }
  })
})

// ---------------------------------------------------------------------------
// Comprehensive lifecycle / branch coverage for the store wrapper.
// Each block targets an action, getter, or conditional branch in
// graphHandoff.ts that the telemetry suite above does not exercise.
// ---------------------------------------------------------------------------

describe('useGraphHandoffStore — initial state + getters', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('starts idle / quiescent with empty pending + generation 0', () => {
    const store = useGraphHandoffStore()
    expect(store.state).toBe('idle')
    expect(store.pending).toBeNull()
    expect(store.generation).toBe(0)
    expect(store.lastResult).toBeNull()
    expect(store.lastInvariant).toBeNull()
    expect(store.isQuiescent).toBe(true)
  })

  it('isQuiescent is true in ready and false mid-pipeline', () => {
    const store = useGraphHandoffStore()
    // A cross-surface handoff lands in loading_fetch → not quiescent.
    store.handoffRequested(envelope())
    expect(store.state).toBe('loading_fetch')
    expect(store.isQuiescent).toBe(false)
    // Drive to ready via recordApplied.
    store.recordApplied('g:episode:abc')
    expect(store.state).toBe('ready')
    expect(store.isQuiescent).toBe(true)
  })

  it('isStale reflects the current generation token', () => {
    const store = useGraphHandoffStore()
    expect(store.isStale(0)).toBe(false) // generation starts at 0
    store.handoffRequested(envelope())
    const g = store.generation
    expect(store.isStale(g)).toBe(false)
    expect(store.isStale(g - 1)).toBe(true)
  })
})

describe('useGraphHandoffStore — reactive mirrors + generation bumps', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('handoffRequested stamps pending, bumps generation, syncs state', () => {
    const store = useGraphHandoffStore()
    const disp = store.handoffRequested(envelope({ cyId: 'g:episode:one' }))
    expect(disp.kind).toBe('accept')
    expect(store.generation).toBe(1)
    expect(store.pending?.cyId).toBe('g:episode:one')
    expect(store.pending?.generation).toBe(1)
    expect(store.state).toBe('loading_fetch')
  })

  it('canvasTapped jumps straight to applying', () => {
    const store = useGraphHandoffStore()
    const disp = store.canvasTapped(
      envelope({ kind: 'graph-node', source: 'canvas-tap', cyId: 'g:node:1' }),
    )
    expect(disp.kind).toBe('accept')
    expect(store.state).toBe('applying')
  })

  it('a second handoffRequested supersedes and bumps generation again', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope({ cyId: 'first' }))
    expect(store.generation).toBe(1)
    const disp = store.handoffRequested(envelope({ cyId: 'second' }))
    expect(store.generation).toBe(2)
    expect(disp.kind === 'accept' && disp.supersededInFlight).toBe(true)
    expect(store.pending?.cyId).toBe('second')
  })
})

describe('useGraphHandoffStore — queue dispositions', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('canvasTapped on the same in-flight target queues (no generation bump)', () => {
    const store = useGraphHandoffStore()
    // Put a graph-node handoff in flight via handoffRequested first.
    store.handoffRequested(envelope({ kind: 'graph-node', cyId: 'g:node:same' }))
    const genBefore = store.generation
    const disp = store.canvasTapped(
      envelope({ kind: 'graph-node', source: 'canvas-tap', cyId: 'g:node:same' }),
    )
    expect(disp.kind).toBe('queue')
    expect(store.generation).toBe(genBefore)
  })

  it('canvasTapped on a different target while in flight supersedes', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope({ kind: 'graph-node', cyId: 'g:node:a' }))
    const disp = store.canvasTapped(
      envelope({ kind: 'graph-node', source: 'canvas-tap', cyId: 'g:node:b' }),
    )
    expect(disp.kind).toBe('accept')
    expect(store.state).toBe('applying')
    expect(store.pending?.cyId).toBe('g:node:b')
  })

  it('expansionRequested while in flight always queues', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope())
    const disp = store.expansionRequested(
      envelope({ source: 'double-tap-expand', camera: { kind: 'preserve' } }),
    )
    expect(disp.kind).toBe('queue')
  })

  it('expansionRequested from quiescent accepts and loads', () => {
    const store = useGraphHandoffStore()
    const disp = store.expansionRequested(
      envelope({
        kind: 'graph-node',
        source: 'double-tap-expand',
        cyId: 'g:node:x',
        camera: { kind: 'preserve' },
      }),
    )
    expect(disp.kind).toBe('accept')
    expect(store.state).toBe('loading_fetch')
  })
})

describe('useGraphHandoffStore — validation failures', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('handoffRequested with whitespace-only ids fails and sets lastResult', () => {
    const store = useGraphHandoffStore()
    const disp = store.handoffRequested(
      envelope({ kind: 'episode', cyId: '   ', metadataPath: '', episodeId: '' }),
    )
    expect(disp.kind).toBe('fail')
    expect(store.lastResult?.status).toBe('failed')
    expect(store.lastResult?.reason).toMatch(/episode envelope requires/)
    // Failed validation must not advance the FSM.
    expect(store.state).toBe('idle')
    expect(captureSpy).toHaveBeenCalledWith(
      'graph_handoff_failed',
      expect.objectContaining({ event_type: 'handoffRequested' }),
    )
  })

  it('graph-node handoff missing cyId fails', () => {
    const store = useGraphHandoffStore()
    const disp = store.handoffRequested(
      envelope({ kind: 'graph-node', cyId: undefined }),
    )
    expect(disp.kind).toBe('fail')
    expect(store.lastResult?.reason).toMatch(/graph-node envelope requires cyId/)
  })

  it('canvasTapped with invalid envelope fails without queueing', () => {
    const store = useGraphHandoffStore()
    const disp = store.canvasTapped(
      envelope({ kind: 'graph-node', source: 'canvas-tap', cyId: '' }),
    )
    expect(disp.kind).toBe('fail')
  })
})

describe('useGraphHandoffStore — clear / reset paths', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('focusCleared resets to ready, drops pending + lastResult, cancels timer', () => {
    vi.useFakeTimers()
    try {
      const store = useGraphHandoffStore()
      store.handoffRequested(envelope())
      store.recordApplied('g:episode:abc')
      expect(store.lastResult).not.toBeNull()
      const disp = store.focusCleared()
      expect(disp.kind).toBe('accept')
      expect(store.state).toBe('ready')
      expect(store.pending).toBeNull()
      expect(store.lastResult).toBeNull()
      // Stuck timer must have been cancelled — advancing time emits nothing.
      captureSpy.mockClear()
      vi.advanceTimersByTime(20_000)
      expect(
        captureSpy.mock.calls.filter((c) => c[0] === 'graph_handoff_stuck')
          .length,
      ).toBe(0)
    } finally {
      vi.useRealTimers()
    }
  })

  it('focusCleared while a handoff is in flight cancels the stuck timer', () => {
    vi.useFakeTimers()
    try {
      const store = useGraphHandoffStore()
      store.handoffRequested(envelope())
      store.focusCleared()
      captureSpy.mockClear()
      vi.advanceTimersByTime(20_000)
      expect(
        captureSpy.mock.calls.filter((c) => c[0] === 'graph_handoff_stuck')
          .length,
      ).toBe(0)
    } finally {
      vi.useRealTimers()
    }
  })

  it('corpusReloaded resets to idle and clears lastResult', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope())
    store.recordApplied('g:episode:abc')
    expect(store.lastResult).not.toBeNull()
    const disp = store.corpusReloaded()
    expect(disp.kind === 'accept' && disp.supersededInFlight).toBe(true)
    expect(store.state).toBe('idle')
    expect(store.pending).toBeNull()
    expect(store.lastResult).toBeNull()
  })

  it('tabReturned drops when not in ready', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope()) // → loading_fetch, not ready
    const disp = store.tabReturned()
    expect(disp.kind).toBe('drop')
  })

  it('tabReturned accepts (stays ready) when ready and cancels stuck timer', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope())
    store.recordApplied('g:episode:abc') // → ready
    const disp = store.tabReturned()
    expect(disp.kind).toBe('accept')
    expect(store.state).toBe('ready')
  })

  it('handoffFailed clears pending, returns to ready, sets failed result', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope())
    const disp = store.handoffFailed('boom')
    expect(disp.kind).toBe('fail')
    expect(store.state).toBe('ready')
    expect(store.pending).toBeNull()
    expect(store.lastResult).toEqual({ status: 'failed', reason: 'boom' })
  })
})

describe('useGraphHandoffStore — layout pipeline events', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('notifyLayoutStart on applying transitions to redrawing_full', () => {
    const store = useGraphHandoffStore()
    store.canvasTapped(
      envelope({ kind: 'graph-node', source: 'canvas-tap', cyId: 'g:n' }),
    )
    expect(store.state).toBe('applying')
    const disp = store.notifyLayoutStart()
    expect(disp.kind).toBe('accept')
    expect(store.state).toBe('redrawing_full')
  })

  it('notifyLayoutStop drives redrawing_full → applying → ready', () => {
    const store = useGraphHandoffStore()
    store.canvasTapped(
      envelope({ kind: 'graph-node', source: 'canvas-tap', cyId: 'g:n' }),
    )
    store.notifyLayoutStart() // applying → redrawing_full
    const d1 = store.notifyLayoutStop() // redrawing_full → applying
    expect(d1.kind).toBe('accept')
    expect(store.state).toBe('applying')
    const d2 = store.notifyLayoutStop() // applying → ready
    expect(d2.kind).toBe('accept')
    expect(store.state).toBe('ready')
  })

  it('notifyLayoutStop in a non-layout state drops', () => {
    const store = useGraphHandoffStore()
    // idle → layoutstop is meaningless.
    const disp = store.notifyLayoutStop()
    expect(disp.kind).toBe('drop')
  })

  it('notifyLayoutStart in a non-applying state drops', () => {
    const store = useGraphHandoffStore()
    const disp = store.notifyLayoutStart()
    expect(disp.kind).toBe('drop')
  })

  it('activeLayoutCount never goes negative on extra layoutstop', () => {
    const store = useGraphHandoffStore()
    // Two stops without a matching start — must not throw / underflow.
    store.notifyLayoutStop()
    store.notifyLayoutStop()
    expect(store.state).toBe('idle')
  })
})

describe('useGraphHandoffStore — advanceState', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('advances through the valid forward pipeline and syncs the mirror', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope()) // → loading_fetch
    expect(store.advanceState('loading_merge')).toBe('loading_merge')
    expect(store.state).toBe('loading_merge')
    expect(store.advanceState('redrawing_full')).toBe('redrawing_full')
    expect(store.state).toBe('redrawing_full')
  })

  it('returns null and does not mutate state on an invalid advance', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope()) // loading_fetch
    expect(store.advanceState('applying')).toBeNull() // not a forward edge
    expect(store.state).toBe('loading_fetch')
  })
})

describe('useGraphHandoffStore — recordApplied branches', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('records applied with fallbackApplied flag propagated', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope())
    captureSpy.mockClear()
    store.recordApplied('g:episode:fb', true)
    expect(store.lastResult).toEqual({
      status: 'applied',
      appliedCyId: 'g:episode:fb',
      fallbackApplied: true,
    })
    expect(captureSpy).toHaveBeenCalledWith(
      'graph_handoff_applied',
      expect.objectContaining({ fallback_applied: true }),
    )
  })

  it('recordApplied in idle drops without mutating lastResult', () => {
    const store = useGraphHandoffStore()
    // Pre-seed a failed result so we can prove it is preserved.
    store.handoffFailed('prior failure')
    store.corpusReloaded() // → idle, but clears lastResult
    store.handoffFailed('prior failure 2') // ready + failed result
    store.corpusReloaded() // → idle, clears again
    // Now genuinely in idle with no pending; recordApplied must drop.
    expect(store.state).toBe('idle')
    const resultBefore = store.lastResult
    store.recordApplied('g:episode:none')
    expect(store.state).toBe('idle')
    expect(store.lastResult).toBe(resultBefore)
    // No applied telemetry on a dropped recordApplied.
    expect(
      captureSpy.mock.calls.filter((c) => c[0] === 'graph_handoff_applied')
        .length,
    ).toBe(0)
  })

  it('recordApplied is idempotent from ready (no extra applied result mutation issue)', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope())
    store.recordApplied('g:episode:abc') // → ready
    captureSpy.mockClear()
    // Second call from ready: tolerated idempotent no-op transition.
    store.recordApplied('g:episode:abc')
    expect(store.state).toBe('ready')
  })
})

describe('useGraphHandoffStore — onStuck subscription', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('invokes registered listeners when the stuck timer fires', () => {
    vi.useFakeTimers()
    try {
      const store = useGraphHandoffStore()
      const seen: string[] = []
      store.onStuck((env) => seen.push(env.source))
      store.handoffRequested(envelope({ source: 'library' }))
      vi.advanceTimersByTime(15_001)
      expect(seen).toEqual(['library'])
      // FSM forced back to ready with a failed result.
      expect(store.state).toBe('ready')
      expect(store.lastResult?.status).toBe('failed')
      expect(store.lastResult?.appliedCyId).toBe('g:episode:abc')
    } finally {
      vi.useRealTimers()
    }
  })

  it('unsubscribe stops a listener from firing', () => {
    vi.useFakeTimers()
    try {
      const store = useGraphHandoffStore()
      let count = 0
      const off = store.onStuck(() => {
        count += 1
      })
      off()
      store.handoffRequested(envelope())
      vi.advanceTimersByTime(15_001)
      expect(count).toBe(0)
    } finally {
      vi.useRealTimers()
    }
  })

  it('unsubscribing twice is a safe no-op (listener already removed)', () => {
    const store = useGraphHandoffStore()
    const off = store.onStuck(() => {})
    off()
    // Second call: indexOf now returns -1, the splice guard is skipped.
    expect(() => off()).not.toThrow()
  })

  it('a throwing listener does not break FSM state recovery', () => {
    vi.useFakeTimers()
    try {
      const store = useGraphHandoffStore()
      store.onStuck(() => {
        throw new Error('listener boom')
      })
      store.handoffRequested(envelope())
      expect(() => vi.advanceTimersByTime(15_001)).not.toThrow()
      expect(store.state).toBe('ready')
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('useGraphHandoffStore — stuck timer guards', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('early-returns when the envelope generation is already superseded', () => {
    vi.useFakeTimers()
    try {
      const store = useGraphHandoffStore()
      store.handoffRequested(envelope({ cyId: 'first' }))
      // corpusReloaded bumps the generation (→ stale) and resets to `idle`
      // WITHOUT clearing the in-flight stuck timer (the clear-on-ready branch
      // only runs when transitioning into `ready`). So the original timer
      // survives but its envelope generation is now stale.
      store.corpusReloaded()
      captureSpy.mockClear()
      vi.advanceTimersByTime(15_001)
      // The surviving timer fires, hits the stale guard, and emits nothing.
      const stuckCalls = captureSpy.mock.calls.filter(
        (c) => c[0] === 'graph_handoff_stuck',
      )
      expect(stuckCalls.length).toBe(0)
      // State unaffected by the stale timer.
      expect(store.state).toBe('idle')
    } finally {
      vi.useRealTimers()
    }
  })

  it('reschedules instead of failing while a layout is active', () => {
    vi.useFakeTimers()
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      const store = useGraphHandoffStore()
      store.handoffRequested(envelope({ source: 'library' }))
      // Simulate a heavy layout that is still running when the timer fires.
      store.notifyLayoutStart()
      captureSpy.mockClear()
      vi.advanceTimersByTime(15_001) // first window: layout active → reschedule
      expect(
        captureSpy.mock.calls.filter((c) => c[0] === 'graph_handoff_stuck')
          .length,
      ).toBe(0)
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('rescheduling'),
      )
      // Layout finishes; the next window must report stuck.
      store.notifyLayoutStop()
      vi.advanceTimersByTime(15_001)
      expect(
        captureSpy.mock.calls.filter((c) => c[0] === 'graph_handoff_stuck')
          .length,
      ).toBe(1)
    } finally {
      warnSpy.mockRestore()
      vi.useRealTimers()
    }
  })
})

describe('useGraphHandoffStore — recordInvariant', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    captureSpy.mockClear()
  })

  it('records an in-sync snapshot (both arrays empty)', () => {
    const store = useGraphHandoffStore()
    store.recordInvariant([], [])
    expect(store.lastInvariant?.missing).toEqual([])
    expect(store.lastInvariant?.extra).toEqual([])
    expect(typeof store.lastInvariant?.ts).toBe('number')
  })

  it('records a divergent snapshot with missing + extra nodes', () => {
    const store = useGraphHandoffStore()
    store.recordInvariant(['g:a'], ['g:b', 'g:c'])
    expect(store.lastInvariant?.missing).toEqual(['g:a'])
    expect(store.lastInvariant?.extra).toEqual(['g:b', 'g:c'])
  })
})

describe('useGraphHandoffStore — dev hooks (import.meta.env.DEV)', () => {
  beforeEach(() => {
    // Force the dev-only window hook / history / event-log / sessionStorage
    // branches (gated on `import.meta.env.DEV`) to execute. The store stamps
    // these at creation time, so DEV must be true before the first call.
    vi.stubEnv('DEV', true)
    setActivePinia(createPinia())
    captureSpy.mockClear()
    // Reset dev globals the store appends to across tests.
    const w = window as unknown as Record<string, unknown>
    delete w.__GIKG_FSM__
    delete w.__GIKG_HANDOFF_STORE__
    delete w.__GIKG_FSM_STATE_HISTORY__
    delete w.__GIKG_FSM_EVENT_LOG__
    window.sessionStorage.clear()
  })

  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('stamps window.__GIKG_FSM__ with live getters', () => {
    const store = useGraphHandoffStore()
    const hook = (window as unknown as { __GIKG_FSM__: Record<string, unknown> })
      .__GIKG_FSM__
    expect(hook).toBeDefined()
    expect(hook.state).toBe('idle')
    expect(hook.pending).toBeNull()
    expect(hook.generation).toBe(0)
    expect(hook.lastResult).toBeNull()

    store.handoffRequested(envelope())
    expect(hook.state).toBe('loading_fetch')
    expect(hook.generation).toBe(1)
    expect((hook.pending as { cyId?: string })?.cyId).toBe('g:episode:abc')
  })

  it('stamps window.__GIKG_HANDOFF_STORE__ with the event API incl clearLastResult', () => {
    const store = useGraphHandoffStore()
    const hook = (
      window as unknown as {
        __GIKG_HANDOFF_STORE__: { clearLastResult: () => void }
      }
    ).__GIKG_HANDOFF_STORE__
    expect(hook).toBeDefined()
    expect(typeof hook.clearLastResult).toBe('function')
    store.handoffFailed('x')
    expect(store.lastResult).not.toBeNull()
    hook.clearLastResult()
    expect(store.lastResult).toBeNull()
  })

  it('captures the state-transition history and event log', () => {
    const store = useGraphHandoffStore()
    store.handoffRequested(envelope())
    store.recordApplied('g:episode:abc')
    const w = window as unknown as {
      __GIKG_FSM_STATE_HISTORY__: string[]
      __GIKG_FSM_EVENT_LOG__: { type: string }[]
    }
    expect(w.__GIKG_FSM_STATE_HISTORY__).toContain('loading_fetch')
    expect(w.__GIKG_FSM_STATE_HISTORY__).toContain('ready')
    expect(w.__GIKG_FSM_EVENT_LOG__.map((e) => e.type)).toContain(
      'handoffRequested',
    )
  })

  it('persists recordInvariant snapshot to sessionStorage', () => {
    const store = useGraphHandoffStore()
    store.recordInvariant(['g:a'], [])
    const raw = window.sessionStorage.getItem('__GIKG_FSM_LAST_INVARIANT__')
    expect(raw).toBeTruthy()
    const parsed = JSON.parse(raw!) as { missing: string[]; extra: string[] }
    expect(parsed.missing).toEqual(['g:a'])
    expect(parsed.extra).toEqual([])
    // __GIKG_FSM__.lastInvariant reads back the module-level snapshot.
    const hook = (
      window as unknown as {
        __GIKG_FSM__: { lastInvariant: { missing: string[] } }
      }
    ).__GIKG_FSM__
    expect(hook.lastInvariant.missing).toEqual(['g:a'])
  })
})
