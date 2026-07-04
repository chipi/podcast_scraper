import { defineStore } from 'pinia'
import { ref } from 'vue'
import { postGraphEvents } from '../api/graphAnalyticsApi'

/**
 * Graph analytics emitter. Components + stores call ``track(action, payload)`` on graph usage
 * (taps, navigation, trail loads, re-centres, search), size dynamics (redraw node/edge counts), and
 * breakage (handoff failures). Events are buffered client-side and flushed in batches — on a timer,
 * when the buffer fills, and (wired by App.vue) on tab-hide / unload — so the network isn't spammed
 * per interaction. Each event is stamped with a client-side millisecond ``ts`` so per-event timing
 * (the "dynamics") survives batching. Best-effort throughout: a failed post is swallowed.
 */

interface GraphEvent {
  action: string
  ts: number
  session_id: string
  [key: string]: unknown
}

const FLUSH_MS = 10_000
const MAX_BUFFER = 200

/** One id per app-load session, stamped on every event so a session's steps can be reconstructed
 *  (and replayed) in order. */
function newSessionId(): string {
  const c = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto
  return c?.randomUUID ? c.randomUUID() : `s_${Math.random().toString(36).slice(2)}`
}

export const useGraphAnalyticsStore = defineStore('graphAnalytics', () => {
  const buffer = ref<GraphEvent[]>([])
  const sessionId = newSessionId()
  let flushTimer: ReturnType<typeof setTimeout> | null = null

  function scheduleFlush(): void {
    if (flushTimer != null) {
      return
    }
    flushTimer = setTimeout(() => {
      flushTimer = null
      flush()
    }, FLUSH_MS)
  }

  function track(action: string, payload: Record<string, unknown> = {}): void {
    if (!action) {
      return
    }
    buffer.value.push({ action, ...payload, session_id: sessionId, ts: Date.now() })
    if (buffer.value.length >= MAX_BUFFER) {
      flush()
    } else {
      scheduleFlush()
    }
  }

  /** Post + clear the buffer now (idempotent when empty). */
  function flush(): void {
    if (flushTimer != null) {
      clearTimeout(flushTimer)
      flushTimer = null
    }
    if (!buffer.value.length) {
      return
    }
    const batch = buffer.value
    buffer.value = []
    postGraphEvents(batch)
  }

  return { buffer, sessionId, track, flush }
})
