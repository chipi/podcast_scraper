import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useGraphNavigationStore } from './graphNavigation'

export interface ReplayEvent {
  action: string
  [key: string]: unknown
}

export type ReplayMode = 'step' | 'timed' | 'scrub'

const TIMED_MS = 800

/**
 * Graph log-replay (#6 analytics). Holds a loaded session's events + a step cursor, and
 * *reconstructs* the graph state at any step from the captured node ids — deterministic, so
 * step / back / scrub / timed-play all work. Reconstruction reuses the live stores (the D#6 trail
 * + focus), so a replayed step drives the real graph exactly as the navigation did: the trail is
 * the rail-nav targets up to the cursor (a re-centre resets it), the focus is the current node, and
 * the graph grows/shrinks as nodes are added/pruned. ``active`` gates the REPLAY banner + a
 * live-interaction lock in the graph.
 */
export const useGraphReplayStore = defineStore('graphReplay', () => {
  const nav = useGraphNavigationStore()

  const events = ref<ReplayEvent[]>([])
  const sessionId = ref<string | null>(null)
  const step = ref(0) // number of events applied: 0..total
  const playing = ref(false)
  const mode = ref<ReplayMode>('step')
  let timer: ReturnType<typeof setInterval> | null = null

  const active = computed(() => sessionId.value !== null)
  const total = computed(() => events.value.length)
  const currentEvent = computed<ReplayEvent | null>(() =>
    step.value > 0 ? events.value[step.value - 1] ?? null : null,
  )

  /** Reconstruct + apply the graph state at the cursor from events[0..step). */
  function apply(): void {
    const trail: string[] = []
    let focus: string | null = null
    for (const e of events.value.slice(0, step.value)) {
      if (e.action === 'graph_recenter' && typeof e.target_id === 'string') {
        trail.length = 0 // a re-centre starts a fresh trail
        focus = e.target_id
      } else if (e.action === 'graph_rail_nav' && typeof e.to_id === 'string') {
        trail.push(e.to_id)
        focus = e.to_id
      } else if (e.action === 'graph_node_tap' && typeof e.id === 'string') {
        focus = e.id
      }
    }
    nav.setTrail(trail)
    if (focus) {
      nav.requestFocusNode(focus)
    }
  }

  function setStep(n: number): void {
    step.value = Math.max(0, Math.min(total.value, n))
    apply()
  }
  function next(): void {
    setStep(step.value + 1)
  }
  function prev(): void {
    setStep(step.value - 1)
  }

  function pause(): void {
    if (timer != null) {
      clearInterval(timer)
      timer = null
    }
    playing.value = false
  }

  function play(): void {
    if (playing.value || !active.value) {
      return
    }
    playing.value = true
    timer = setInterval(() => {
      if (step.value >= total.value) {
        pause()
        return
      }
      next()
    }, TIMED_MS)
  }

  function load(id: string, evs: ReplayEvent[]): void {
    pause()
    sessionId.value = id
    events.value = evs
    step.value = 0
    mode.value = 'step'
    apply()
  }

  function exit(): void {
    pause()
    sessionId.value = null
    events.value = []
    step.value = 0
    nav.clearTrail()
  }

  return {
    events,
    sessionId,
    step,
    playing,
    mode,
    active,
    total,
    currentEvent,
    load,
    exit,
    setStep,
    next,
    prev,
    play,
    pause,
  }
})
