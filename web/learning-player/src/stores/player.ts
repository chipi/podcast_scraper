import { defineStore } from 'pinia'
import { ref } from 'vue'
import { PLAYBACK_RATES } from '../player/transcriptSync'

/**
 * Player store — the single source of truth for audio playback state + transport (#1307).
 *
 * Playback used to live as local refs in PlayerView; MediaSession (#1308), a notification, and the
 * UI must all reflect ONE state, and native controls outlive any single view — so it lives here.
 *
 * The `<audio>` element is still rendered by PlayerView (kept in the DOM so the app's transport,
 * the transcript-follow, and the e2e all see a real element) and registered via `bind()`; the store
 * owns the STATE + the transport actions that operate on it. Deep-link/resume seeking, playback
 * persistence, queue-advance and the transcript sync-offset stay in the view (they're
 * view/router/episode concerns) and drive the store through these setters/actions.
 *
 * A later step (native background audio / mini-player, #1310) can promote the element to a
 * store-owned detached `Audio()` for cross-view persistence; the state contract here stays the same.
 */
export const usePlayerStore = defineStore('player', () => {
  const el = ref<HTMLAudioElement | null>(null)
  const playing = ref(false)
  const currentTime = ref(0)
  const duration = ref(0)
  const rate = ref(1)
  const audioError = ref(false)

  /** Register (or clear) the DOM `<audio>` element the store drives. */
  function bind(element: HTMLAudioElement | null): void {
    el.value = element
    if (element) element.playbackRate = rate.value
  }

  // --- element event sinks (PlayerView's <audio> forwards these) ---
  function onPlay(): void {
    playing.value = true
  }
  function onPause(): void {
    playing.value = false
  }
  function onTimeUpdate(): void {
    currentTime.value = el.value?.currentTime ?? 0
  }
  function onDurationChange(): void {
    duration.value = el.value?.duration || 0
  }
  function onError(): void {
    audioError.value = true
  }
  /** New episode loading — clear transient state (source swap happens in the view). */
  function resetForLoad(): void {
    playing.value = false
    currentTime.value = 0
    duration.value = 0
    audioError.value = false
  }

  // --- transport (MediaSession + UI both call these) ---
  function toggle(): void {
    const e = el.value
    if (!e) return
    if (e.paused) void e.play()
    else e.pause()
  }
  function seek(to: number): void {
    const e = el.value
    if (!e) return
    e.currentTime = Math.max(0, Math.min(to, duration.value || to))
  }
  function skip(delta: number): void {
    seek(currentTime.value + delta)
  }
  function setRate(r: number): void {
    rate.value = r
    if (el.value) el.value.playbackRate = r
  }
  function cycleRate(): void {
    const i = PLAYBACK_RATES.indexOf(rate.value as (typeof PLAYBACK_RATES)[number])
    setRate(PLAYBACK_RATES[(i + 1) % PLAYBACK_RATES.length])
  }

  return {
    el,
    playing,
    currentTime,
    duration,
    rate,
    audioError,
    bind,
    onPlay,
    onPause,
    onTimeUpdate,
    onDurationChange,
    onError,
    resetForLoad,
    toggle,
    seek,
    skip,
    setRate,
    cycleRate,
  }
})
