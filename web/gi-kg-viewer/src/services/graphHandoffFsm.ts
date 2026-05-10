/**
 * Pure finite state machine for the graph handoff orchestrator.
 *
 * Implements the FSM design locked in the plan
 * (`/Users/markodragoljevic/.claude/plans/in-this-b-tanch-gentle-pillow.md`,
 * §"FSM design specification (locked)") and pressure-tested in subsequent
 * passes.
 *
 *   8 states: idle → loading_fetch → loading_bootstrap → loading_merge →
 *             redrawing_incremental | redrawing_full → applying → ready
 *
 *   9 events: handoffRequested, canvasTapped, expansionRequested, focusCleared,
 *             tabReturned, corpusReloaded, handoffFailed, layoutstop, layoutstart
 *
 * Pure module — no Vue, no Pinia, no Cytoscape, no `setTimeout` for
 * synchronization (timeouts are allowed; see decision #16 / `STUCK_TIMEOUT_MS`).
 * Easy to unit-test (see `graphHandoffFsm.test.ts`).
 *
 * The Pinia store at `stores/graphHandoff.ts` wraps this module and provides
 * reactive bindings + dev-only `window.__GIKG_FSM__`. Call sites should not
 * import from this file directly — go through the store.
 *
 * # Bare-await contract (F3e — review-required hygiene rule)
 *
 * Every `await` in this module's runtime callers (notably orchestrator code in
 * `GraphCanvas.vue`'s `loadEpisodeSliceForTerritoryStrip` and any future
 * orchestrator helpers) MUST be paired with an `isStale(envelope.generation)`
 * check on the next non-empty line. Without the stale-check, an in-flight
 * handoff can complete its async work and write UI state for an envelope that
 * has already been superseded by a newer click — producing the exact
 * "old episode wins after rapid clicks" race the FSM is designed to prevent.
 *
 * The 8 documented check points in this module's runtime are:
 *
 *   1. Before `await fetchCorpusEpisodeDetail`
 *   2. After `fetchCorpusEpisodeDetail`, before any `subject.set*` write
 *   3. Before `await artifacts.appendRelativeArtifacts`
 *   4. After `appendRelativeArtifacts`, before any Pinia write
 *   5. In `layoutstop` listener, before `finishLayoutPass`
 *   6. At entry of `tryApplyPendingFocus`
 *   7. Before `cy.animate` and inside its `complete` callback
 *   8. Inside `nextTick` chains in lifecycle hooks (`onMounted` / `onActivated`)
 *
 * Sites 1–4 are inside `loadEpisodeSliceForTerritoryStrip`; sites 5–7 are wired
 * in F3a; site 8 is wired by the watcher rework in F3c. Adding a new `await`
 * to an orchestrator path REQUIRES adding a matching `isStale()` check — this
 * file's contract.
 *
 * Code review checklist for any PR touching the orchestrator:
 *   - [ ] every new `await` in an orchestrator helper has a paired `isStale()`
 *   - [ ] the stale-check returns early without mutating UI state
 *   - [ ] generation is captured at function entry (not re-read after await)
 *
 * A custom ESLint rule could enforce this mechanically; tracked as a future
 * improvement. For now, this contract is review-only.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** All FSM states. */
export type FsmState =
  | 'idle'
  | 'loading_fetch'
  | 'loading_bootstrap'
  | 'loading_merge'
  | 'redrawing_incremental'
  | 'redrawing_full'
  | 'applying'
  | 'ready'

/** Subject kinds the FSM can target. ``person`` is reserved for 2.7 (decision #12). */
export type EnvelopeKind = 'episode' | 'graph-node' | 'topic'

/** Load-source enum — medium granularity per decision #2. */
export type LoadSource = 'subject-external' | 'digest-external' | 'graph-internal'

/**
 * Originating surface of the envelope. New surfaces add a string here, not a
 * branch elsewhere. See `HANDOFF_MATRIX.md` § entry points.
 */
export type EnvelopeSource =
  | 'library'
  | 'digest'
  | 'search'
  | 'dashboard'
  | 'episode-panel'
  | 'node-detail'
  | 'subject-rail'
  | 'status-bar'
  | 'explore'
  | 'tab-switch'
  | 'canvas-tap'
  | 'minimap'
  | 'double-tap-expand'
  | 'restore-preference'
  | 'corpus-reload'

/** Camera strategy on envelope (decision #11). */
export type CameraStrategy =
  | { kind: 'center'; cyId: string; includes?: string[] }
  /**
   * Center on whatever primary cyId the orchestrator resolves from
   * ``cyId | metadataPath | episodeId``. Used by handoffs that don't know the
   * cyId at click time (Library row, Episode panel "Open in graph", etc.).
   */
  | { kind: 'center-on-target'; includes?: string[] }
  | { kind: 'fit' }
  | { kind: 'preserve' }
  | { kind: 'none' }

/**
 * Single canonical handoff envelope. All entry points build one of these
 * (decision #4). The FSM owns the ``generation`` field — callers must not set it.
 */
export type GraphHandoffEnvelope = {
  kind: EnvelopeKind
  cyId?: string
  metadataPath?: string
  episodeId?: string
  source: EnvelopeSource
  loadSource: LoadSource
  camera: CameraStrategy
  highlights?: string[]
  /**
   * ``suppressCamera`` — opt-out from the camera strategy. Today only set by
   * mini-map clicks (decision #6).
   */
  suppressCamera?: boolean
  generation: number
}

/** Caller-facing envelope (no generation; FSM stamps it). */
export type EnvelopeInput = Omit<GraphHandoffEnvelope, 'generation'>

/** Outcome of a handoff request. */
export type HandoffResultStatus = 'applied' | 'failed' | 'superseded'

export type HandoffResult = {
  status: HandoffResultStatus
  reason?: string
  appliedCyId?: string
  fallbackApplied?: boolean
}

/** First-class FSM events. */
export type FsmEvent =
  | { type: 'handoffRequested'; envelope: EnvelopeInput }
  | { type: 'canvasTapped'; envelope: EnvelopeInput }
  | { type: 'expansionRequested'; envelope: EnvelopeInput }
  | { type: 'focusCleared' }
  | { type: 'tabReturned' }
  | { type: 'corpusReloaded' }
  | { type: 'handoffFailed'; reason: string }
  | { type: 'layoutstop' }
  | { type: 'layoutstart' }

/**
 * Disposition of an event from the FSM's perspective. The orchestrator runtime
 * (in the Pinia store) reads this to decide whether to launch async work,
 * cancel in-flight work, or no-op.
 */
export type EventDisposition =
  /** Transition state, accept envelope (if any), launch any required work. */
  | {
      kind: 'accept'
      nextState: FsmState
      envelope: GraphHandoffEnvelope | null
      /** Generation-bumping events set this to true; the runtime cancels in-flight work. */
      supersededInFlight: boolean
    }
  /** Same envelope already in flight; coalesce / let the in-flight finish. */
  | { kind: 'queue'; reason: string }
  /** Event is meaningless in current state; ignore. */
  | { kind: 'drop'; reason: string }
  /** Event indicates failure; transition to ready with logged failure. */
  | { kind: 'fail'; reason: string }

// ---------------------------------------------------------------------------
// FSM core
// ---------------------------------------------------------------------------

/**
 * Stuck-handoff timeout in milliseconds (decision #16). The "no time-based
 * gates" rule (concern #3) applies to *synchronization*; *timeouts* are
 * allowed and necessary — without one, an in-flight handoff that never
 * resolves stays stuck forever.
 */
export const STUCK_TIMEOUT_MS = 5000

/**
 * Internal mutable FSM state. Constructed via {@link createFsm}.
 */
export type Fsm = {
  /** Current state. Read-only externally (use {@link applyEvent}). */
  state: FsmState
  /** In-flight envelope or null when idle/ready. */
  pending: GraphHandoffEnvelope | null
  /** Monotonic generation counter; bumped on supersession. */
  generation: number
}

export function createFsm(): Fsm {
  return {
    state: 'idle',
    pending: null,
    generation: 0,
  }
}

/**
 * Validate an envelope's required fields per kind. Returns null on success or
 * a reason string on failure. Pure function — no side effects.
 */
export function validateEnvelope(env: EnvelopeInput): string | null {
  if (env.kind === 'episode') {
    const hasId = (env.cyId && env.cyId.trim()) ||
      (env.metadataPath && env.metadataPath.trim()) ||
      (env.episodeId && env.episodeId.trim())
    if (!hasId) {
      return 'episode envelope requires at least one of cyId / metadataPath / episodeId'
    }
  }
  if (env.kind === 'graph-node' && !(env.cyId && env.cyId.trim())) {
    return 'graph-node envelope requires cyId'
  }
  if (env.kind === 'topic' && !(env.cyId && env.cyId.trim())) {
    return 'topic envelope requires cyId'
  }
  if (env.camera.kind === 'center' && !env.camera.cyId.trim()) {
    return "camera strategy 'center' requires cyId"
  }
  return null
}

/**
 * Whether the new envelope targets the same node as the in-flight one. Used
 * by the canvasTapped re-entrance policy ("queue same-target, supersede
 * different-target").
 */
function envelopesShareTarget(
  a: GraphHandoffEnvelope | null,
  b: EnvelopeInput,
): boolean {
  if (!a) return false
  if (a.kind !== b.kind) return false
  if (a.cyId && b.cyId) return a.cyId === b.cyId
  if (a.metadataPath && b.metadataPath) return a.metadataPath === b.metadataPath
  if (a.episodeId && b.episodeId) return a.episodeId === b.episodeId
  return false
}

/** What state to enter when starting a fresh handoff from idle/ready. */
function startStateForEnvelope(env: EnvelopeInput): FsmState {
  // Direct selection (canvas tap / mini-map): no load, no redraw, jump to apply.
  if (
    env.source === 'canvas-tap' ||
    env.source === 'minimap'
  ) {
    return 'applying'
  }
  // Expansion: load (artifacts), then redraw_incremental, then apply.
  if (env.source === 'double-tap-expand' || env.source === 'node-detail') {
    return 'loading_fetch'
  }
  // Cross-surface handoff: full pipeline.
  return 'loading_fetch'
}

/**
 * Apply an event to the FSM. Returns the disposition for the runtime to act on,
 * AND mutates the FSM in place.
 *
 * Re-entrance policy (per decision #5 / FSM spec):
 * - `handoffRequested`: always supersede.
 * - `canvasTapped`: supersede different-target / queue same-target.
 * - `expansionRequested`: always queue.
 * - `focusCleared`: always supersede with empty envelope.
 * - `tabReturned`: drop if not in `ready`.
 * - `corpusReloaded`: always full reset.
 */
export function applyEvent(fsm: Fsm, event: FsmEvent): EventDisposition {
  switch (event.type) {
    case 'handoffRequested': {
      const err = validateEnvelope(event.envelope)
      if (err) return { kind: 'fail', reason: err }
      const wasInFlight = fsm.state !== 'idle' && fsm.state !== 'ready'
      fsm.generation += 1
      const stamped: GraphHandoffEnvelope = {
        ...event.envelope,
        generation: fsm.generation,
      }
      fsm.pending = stamped
      fsm.state = startStateForEnvelope(event.envelope)
      return {
        kind: 'accept',
        nextState: fsm.state,
        envelope: stamped,
        supersededInFlight: wasInFlight,
      }
    }
    case 'canvasTapped': {
      const err = validateEnvelope(event.envelope)
      if (err) return { kind: 'fail', reason: err }
      // Queue if same target as in-flight (idempotent click).
      if (
        fsm.state !== 'idle' &&
        fsm.state !== 'ready' &&
        envelopesShareTarget(fsm.pending, event.envelope)
      ) {
        return { kind: 'queue', reason: 'same target as in-flight envelope' }
      }
      const wasInFlight = fsm.state !== 'idle' && fsm.state !== 'ready'
      fsm.generation += 1
      const stamped: GraphHandoffEnvelope = {
        ...event.envelope,
        generation: fsm.generation,
      }
      fsm.pending = stamped
      fsm.state = 'applying' // canvas taps skip load + redraw barriers
      return {
        kind: 'accept',
        nextState: fsm.state,
        envelope: stamped,
        supersededInFlight: wasInFlight,
      }
    }
    case 'expansionRequested': {
      const err = validateEnvelope(event.envelope)
      if (err) return { kind: 'fail', reason: err }
      // Always queue if anything is in flight (additive; cancelling loses work).
      if (fsm.state !== 'idle' && fsm.state !== 'ready') {
        return { kind: 'queue', reason: 'expansion is additive; queueing' }
      }
      fsm.generation += 1
      const stamped: GraphHandoffEnvelope = {
        ...event.envelope,
        generation: fsm.generation,
      }
      fsm.pending = stamped
      fsm.state = 'loading_fetch'
      return {
        kind: 'accept',
        nextState: fsm.state,
        envelope: stamped,
        supersededInFlight: false,
      }
    }
    case 'focusCleared': {
      const wasInFlight = fsm.state !== 'idle' && fsm.state !== 'ready'
      fsm.generation += 1
      fsm.pending = null
      fsm.state = 'ready'
      return {
        kind: 'accept',
        nextState: fsm.state,
        envelope: null,
        supersededInFlight: wasInFlight,
      }
    }
    case 'tabReturned': {
      // Reconcile-only (decision #7). Drop if not ready.
      if (fsm.state !== 'ready') {
        return { kind: 'drop', reason: `not ready (state=${fsm.state})` }
      }
      // Stay in ready; the runtime runs the reconcile predicate.
      return {
        kind: 'accept',
        nextState: 'ready',
        envelope: null,
        supersededInFlight: false,
      }
    }
    case 'corpusReloaded': {
      fsm.generation += 1
      fsm.pending = null
      fsm.state = 'idle'
      return {
        kind: 'accept',
        nextState: 'idle',
        envelope: null,
        supersededInFlight: true,
      }
    }
    case 'handoffFailed': {
      // Preserve previous selection by clearing pending without changing
      // selection state on the runtime side (decision #15 — graceful degrade).
      fsm.pending = null
      fsm.state = 'ready'
      return { kind: 'fail', reason: event.reason }
    }
    case 'layoutstop': {
      // Drives redrawing_* → applying.
      if (fsm.state === 'redrawing_incremental' || fsm.state === 'redrawing_full') {
        fsm.state = 'applying'
        return {
          kind: 'accept',
          nextState: 'applying',
          envelope: fsm.pending,
          supersededInFlight: false,
        }
      }
      // applying → ready (apply phase complete).
      if (fsm.state === 'applying') {
        fsm.state = 'ready'
        return {
          kind: 'accept',
          nextState: 'ready',
          envelope: fsm.pending,
          supersededInFlight: false,
        }
      }
      return { kind: 'drop', reason: `layoutstop ignored in state=${fsm.state}` }
    }
    case 'layoutstart': {
      // applying → redrawing_full when a force-redraw fires mid-apply.
      if (fsm.state === 'applying') {
        fsm.state = 'redrawing_full'
        return {
          kind: 'accept',
          nextState: 'redrawing_full',
          envelope: fsm.pending,
          supersededInFlight: false,
        }
      }
      return { kind: 'drop', reason: `layoutstart ignored in state=${fsm.state}` }
    }
  }
}

/**
 * Test whether a stamped envelope's generation is still current. Use after
 * every async await before mutating UI state (decision #4 / 8+ check points).
 */
export function isStale(fsm: Fsm, envelopeGeneration: number): boolean {
  return envelopeGeneration !== fsm.generation
}

/**
 * Advance the FSM through internal state transitions that don't take an
 * external event. Used by the orchestrator runtime to advance from
 * `loading_fetch → loading_bootstrap → loading_merge → redrawing_*` once the
 * relevant async work resolves.
 *
 * Returns the new state, or null if the transition is invalid (caller error).
 */
export function advanceState(fsm: Fsm, target: FsmState): FsmState | null {
  const valid = isValidAdvance(fsm.state, target)
  if (!valid) return null
  fsm.state = target
  return target
}

const FORWARD_TRANSITIONS: Record<FsmState, FsmState[]> = {
  idle: ['loading_fetch', 'applying', 'ready'],
  loading_fetch: ['loading_bootstrap', 'loading_merge', 'ready'],
  loading_bootstrap: ['loading_merge', 'ready'],
  loading_merge: ['redrawing_incremental', 'redrawing_full', 'ready'],
  redrawing_incremental: ['applying', 'ready'],
  redrawing_full: ['applying', 'ready'],
  applying: ['ready', 'redrawing_full'],
  ready: ['loading_fetch', 'applying', 'idle'],
}

export function isValidAdvance(from: FsmState, to: FsmState): boolean {
  return (FORWARD_TRANSITIONS[from] ?? []).includes(to)
}
