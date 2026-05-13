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
import { beforeEach, describe, expect, it, vi } from 'vitest'

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
