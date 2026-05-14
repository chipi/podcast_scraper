/**
 * Pinia store wrapping the pure FSM (`services/graphHandoffFsm.ts`) for the
 * graph handoff orchestrator.
 *
 * Responsibilities:
 *   - Hold the FSM as reactive state for component subscribers.
 *   - Stamp the dev-only `window.__GIKG_FSM__` for E2E inspection (matches
 *     `__GIKG_CY_DEV__` pattern).
 *   - Implement the 15s wall-clock stuck-handoff detector (decision #16).
 *   - Provide the public API (handoffRequested, canvasTapped, etc.) that
 *     entry points will migrate to in C5.
 *
 * The store is intentionally *additive* in C4: existing watchers in
 * `GraphCanvas.vue` continue to drive the canvas. The FSM tracks state in
 * parallel; C5 + C6 progressively make the FSM authoritative.
 */

import { defineStore } from 'pinia'
import posthog from 'posthog-js'
import { computed, ref } from 'vue'
import {
  advanceState as fsmAdvanceState,
  applyEvent as fsmApplyEvent,
  createFsm,
  isStale as fsmIsStale,
  STUCK_TIMEOUT_MS,
  type EnvelopeInput,
  type EventDisposition,
  type Fsm,
  type FsmEvent,
  type FsmState,
  type GraphHandoffEnvelope,
  type HandoffResult,
} from '../services/graphHandoffFsm'

export type GraphHandoffStuckListener = (envelope: GraphHandoffEnvelope) => void

export const useGraphHandoffStore = defineStore('graphHandoff', () => {
  const fsm: Fsm = createFsm()

  // Reactive mirrors so components / dev tools can `storeToRefs` them.
  const state = ref<FsmState>(fsm.state)
  const pending = ref<GraphHandoffEnvelope | null>(fsm.pending)
  const generation = ref<number>(fsm.generation)

  /** Last `HandoffResult` emitted; consumed by the failure UI strip (C7). */
  const lastResult = ref<HandoffResult | null>(null)

  /** Bookkeeping for the wall-clock stuck-handoff timer. */
  let stuckTimer: ReturnType<typeof setTimeout> | null = null
  const stuckListeners: GraphHandoffStuckListener[] = []

  function syncReactive(): void {
    state.value = fsm.state
    pending.value = fsm.pending
    generation.value = fsm.generation
    // T1 — dev-only state-history capture for state-walking integration tests.
    // Without this, Playwright tests can only observe the FSM's CURRENT state
    // (which is `ready` by the time the test reads it after a handoff settles)
    // — they can't verify the FSM actually walked through intermediate
    // states. The history array captures every transition for later assertion.
    if (typeof window !== 'undefined' && import.meta.env?.DEV) {
      const w = window as unknown as { __GIKG_FSM_STATE_HISTORY__?: string[] }
      if (!w.__GIKG_FSM_STATE_HISTORY__) {
        w.__GIKG_FSM_STATE_HISTORY__ = []
      }
      w.__GIKG_FSM_STATE_HISTORY__.push(fsm.state)
    }
  }

  function clearStuckTimer(): void {
    if (stuckTimer != null) {
      clearTimeout(stuckTimer)
      stuckTimer = null
    }
  }

  function scheduleStuckTimer(envelope: GraphHandoffEnvelope): void {
    clearStuckTimer()
    stuckTimer = setTimeout(() => {
      // Wall-clock fired — handoff did not reach `ready` in time. Per the
      // stuck-detection contract: log + clear + emit listeners; the user can
      // try again and the new generation resolves cleanly.
      if (fsmIsStale(fsm, envelope.generation)) {
        // Already superseded; nothing to do.
        return
      }
      console.warn(
        `[graphHandoff] handoffStuck source=${envelope.source} kind=${envelope.kind} generation=${envelope.generation}`,
      )
      // F6 — telemetry: record the stuck event for monitoring.
      try {
        posthog.capture('graph_handoff_stuck', {
          source: envelope.source,
          kind: envelope.kind,
          load_source: envelope.loadSource,
          generation: envelope.generation,
          timeout_ms: STUCK_TIMEOUT_MS,
        })
      } catch {
        /* telemetry must not affect runtime */
      }
      // Force the FSM back to ready so the next event proceeds normally.
      // Preserve the envelope's intended ``cyId`` on ``appliedCyId`` even
      // though we're reporting `failed`: layout that's still settling in the
      // background may eventually finish and ``GraphCanvas.finishLayoutPass``
      // can use this id to restore selection + camera. Without this hook the
      // user sees an empty selection and fit-all camera after every stuck
      // handoff that eventually completes (rapid digest pills on a heavy
      // KG-second-wave merged graph — see GH #771).
      fsm.pending = null
      fsm.state = 'ready'
      lastResult.value = {
        status: 'failed',
        reason: `stuck-timeout after ${STUCK_TIMEOUT_MS}ms`,
        appliedCyId: envelope.cyId,
      }
      syncReactive()
      for (const fn of stuckListeners) {
        try {
          fn(envelope)
        } catch {
          // Listener errors must not affect FSM state.
        }
      }
    }, STUCK_TIMEOUT_MS)
  }

  /**
   * Apply an event to the FSM and synchronise reactive state. Returns the FSM
   * disposition so the orchestrator runtime can decide what async work to
   * launch / cancel.
   */
  function applyEvent(event: FsmEvent): EventDisposition {
    // T3 — dev-only event log for architectural-invariant contract tests.
    // Pushes every FSM event with its envelope so Playwright contract tests
    // can mechanically verify "surface X fires event Y with source Z" without
    // monkeypatching the store. Production builds: this branch is dead-code
    // -eliminated.
    if (typeof window !== 'undefined' && import.meta.env?.DEV) {
      const w = window as unknown as { __GIKG_FSM_EVENT_LOG__?: FsmEvent[] }
      if (!w.__GIKG_FSM_EVENT_LOG__) {
        w.__GIKG_FSM_EVENT_LOG__ = []
      }
      w.__GIKG_FSM_EVENT_LOG__.push(event)
    }
    const disp = fsmApplyEvent(fsm, event)
    syncReactive()
    if (disp.kind === 'accept' && disp.envelope) {
      // New envelope accepted → arm the stuck timer.
      scheduleStuckTimer(disp.envelope)
      // F6 — telemetry: record the handoff start.
      try {
        posthog.capture('graph_handoff_started', {
          source: disp.envelope.source,
          kind: disp.envelope.kind,
          load_source: disp.envelope.loadSource,
          camera_kind: disp.envelope.camera.kind,
          generation: disp.envelope.generation,
          superseded_in_flight: disp.supersededInFlight,
          event_type: event.type,
        })
      } catch {
        /* telemetry must not affect runtime */
      }
    } else if (
      disp.kind === 'accept' &&
      disp.envelope === null &&
      fsm.state === 'ready'
    ) {
      // Transition into ready with no envelope (focusCleared / tabReturned /
      // corpusReloaded) → cancel any pending stuck timer.
      clearStuckTimer()
    } else if (disp.kind === 'fail') {
      lastResult.value = { status: 'failed', reason: disp.reason }
      clearStuckTimer()
      try {
        posthog.capture('graph_handoff_failed', {
          reason: disp.reason,
          event_type: event.type,
        })
      } catch {
        /* telemetry must not affect runtime */
      }
    }
    return disp
  }

  // -------------------------------------------------------------------------
  // Public event API. Call sites migrate from bespoke triplets
  // (subject.* + nav.requestFocusNode + setLoadSource) to one of these in C5.
  // -------------------------------------------------------------------------

  function handoffRequested(envelope: EnvelopeInput): EventDisposition {
    return applyEvent({ type: 'handoffRequested', envelope })
  }

  function canvasTapped(envelope: EnvelopeInput): EventDisposition {
    return applyEvent({ type: 'canvasTapped', envelope })
  }

  function expansionRequested(envelope: EnvelopeInput): EventDisposition {
    return applyEvent({ type: 'expansionRequested', envelope })
  }

  function focusCleared(): EventDisposition {
    // The Escape contract is "the user wants the prior selection gone."
    // Drop ``lastResult`` so any downstream consumer that restores
    // selection from the FSM's last applied id (e.g. the post-layout
    // selection-restore in ``GraphCanvas.finishLayoutPass``) doesn't
    // re-anchor onto the just-Escaped target on the next layoutstop.
    lastResult.value = null
    return applyEvent({ type: 'focusCleared' })
  }

  function tabReturned(): EventDisposition {
    return applyEvent({ type: 'tabReturned' })
  }

  function corpusReloaded(): EventDisposition {
    // A fresh corpus has no prior selection to preserve. Clear
    // ``lastResult`` for the same reason as ``focusCleared`` above —
    // the FSM's last applied cyId belongs to the *previous* corpus and
    // any restore attempt would either fail (node not in cy) or hit a
    // collision with an unrelated node.
    lastResult.value = null
    return applyEvent({ type: 'corpusReloaded' })
  }

  function handoffFailed(reason: string): EventDisposition {
    return applyEvent({ type: 'handoffFailed', reason })
  }

  function notifyLayoutStart(): EventDisposition {
    return applyEvent({ type: 'layoutstart' })
  }

  function notifyLayoutStop(): EventDisposition {
    return applyEvent({ type: 'layoutstop' })
  }

  /**
   * Test whether a stamped envelope's generation is still current. Use this
   * after any async await inside the orchestrator runtime before mutating
   * UI state. (See decision #4 / FSM spec § generation-token check points.)
   */
  function isStale(envelopeGeneration: number): boolean {
    return fsmIsStale(fsm, envelopeGeneration)
  }

  /**
   * F2 — advance the FSM through internal state transitions during the load →
   * apply pipeline. Used by `GraphCanvas.vue` to drive the FSM through
   * `loading_fetch → loading_merge → redrawing_* → applying → ready` as the
   * runtime work progresses.
   *
   * Returns the new state on a valid advance, or `null` if the transition
   * is invalid (caller error). Reactive mirrors are synchronised on success.
   */
  function advanceState(target: FsmState): FsmState | null {
    const result = fsmAdvanceState(fsm, target)
    if (result !== null) {
      syncReactive()
    }
    return result
  }

  /**
   * Record an `applied` outcome for the current envelope. Also resets the FSM
   * to `ready` and clears the stuck timer. Call this when an entry-point
   * handoff has fully landed (selection applied, camera centred). C5 wires
   * this from `finishLayoutPass`; C6 will replace with a proper FSM
   * apply-phase transition.
   */
  function recordApplied(appliedCyId: string, fallbackApplied = false): void {
    const previousPending = fsm.pending
    lastResult.value = {
      status: 'applied',
      appliedCyId,
      fallbackApplied,
    }
    fsm.pending = null
    fsm.state = 'ready'
    syncReactive()
    clearStuckTimer()
    // F6 — telemetry: record successful handoff completion.
    if (previousPending) {
      try {
        posthog.capture('graph_handoff_applied', {
          source: previousPending.source,
          kind: previousPending.kind,
          load_source: previousPending.loadSource,
          generation: previousPending.generation,
          applied_cy_id: appliedCyId,
          fallback_applied: fallbackApplied,
        })
      } catch {
        /* telemetry must not affect runtime */
      }
    }
  }

  /** Subscribe to stuck-handoff events for telemetry / failure UI. */
  function onStuck(fn: GraphHandoffStuckListener): () => void {
    stuckListeners.push(fn)
    return () => {
      const i = stuckListeners.indexOf(fn)
      if (i >= 0) stuckListeners.splice(i, 1)
    }
  }

  // -------------------------------------------------------------------------
  // Dev hook for E2E specs and devtools
  // -------------------------------------------------------------------------

  if (typeof window !== 'undefined' && import.meta.env?.DEV) {
    ;(window as unknown as { __GIKG_FSM__?: object }).__GIKG_FSM__ = {
      get state() {
        return fsm.state
      },
      get pending() {
        return fsm.pending
      },
      get generation() {
        return fsm.generation
      },
      get lastResult() {
        return lastResult.value
      },
    }
    // T4 / T5 dev-only test hook: expose the store's event methods so
    // Playwright contract tests can trigger FSM events (e.g. `handoffFailed`)
    // without having to set up the full pipeline. Production builds: this
    // branch is dead-code-eliminated by Vite (`import.meta.env.DEV` is a
    // compile-time literal).
    ;(window as unknown as { __GIKG_HANDOFF_STORE__?: object }).__GIKG_HANDOFF_STORE__ = {
      handoffRequested,
      canvasTapped,
      expansionRequested,
      focusCleared,
      tabReturned,
      corpusReloaded,
      handoffFailed,
      recordApplied,
      clearLastResult: () => {
        lastResult.value = null
      },
    }
  }

  /** Computed projection of "is the FSM idle / ready"? Useful for waiters. */
  const isQuiescent = computed(
    () => state.value === 'idle' || state.value === 'ready',
  )

  return {
    state,
    pending,
    generation,
    lastResult,
    isQuiescent,
    handoffRequested,
    canvasTapped,
    expansionRequested,
    focusCleared,
    tabReturned,
    corpusReloaded,
    handoffFailed,
    notifyLayoutStart,
    notifyLayoutStop,
    isStale,
    advanceState,
    recordApplied,
    onStuck,
  }
})
