import { defineStore } from 'pinia'
import { ref } from 'vue'
import { PLAYBACK_RATES } from '../player/transcriptSync'
import { startBackgroundAudio, stopBackgroundAudio } from '../services/native'

/**
 * Player store — the single source of truth for audio playback state + transport (#1307).
 *
 * Playback used to live as local refs in PlayerView; MediaSession (#1308), a notification, and the
 * UI must all reflect ONE state, and native controls outlive any single view — so it lives here.
 *
 * ## The element lives HERE now (#1587)
 *
 * It used to be rendered by PlayerView, which meant navigating away UNMOUNTED it and audio stopped.
 * You could not browse, search or look anything up while listening — and the learning features are
 * most valuable mid-listen (hear a name, want the entity card), so every one of those journeys cost
 * you the thing you were listening to. That single fact suppressed the whole differentiator.
 *
 * The store now owns a detached `Audio()`, exactly as this file's own comment planned for. Nothing
 * about the state contract changed; what changed is who holds the element and therefore how long it
 * lives. `bind()` is gone — the store creates its element lazily on first `load()` (lazily so that
 * merely importing the store in a test or on a page with no audio never constructs one).
 *
 * The view still owns what is genuinely view-shaped: deep-link/resume seeking, playback persistence
 * and the transcript sync-offset. Queue-advance moved here, because it must keep working when no
 * player view is mounted.
 */
export const usePlayerStore = defineStore('player', () => {
  const el = ref<HTMLAudioElement | null>(null)
  /** The episode currently loaded into the element — drives the mini-player and queue-advance. */
  const currentSlug = ref<string | null>(null)
  const currentTitle = ref<string | null>(null)
  const currentArtwork = ref<string | null>(null)
  const playing = ref(false)
  const currentTime = ref(0)
  const duration = ref(0)
  const rate = ref(1)
  const audioError = ref(false)

  /**
   * The store's own detached element, created on first use and never torn down — that persistence
   * IS the feature. Listeners are attached once, here, rather than by a template that unmounts.
   */
  function ensureElement(): HTMLAudioElement {
    if (el.value) return el.value
    const audio = new Audio()
    audio.preload = 'metadata'
    // Attached to <body>, not left detached. Detached would work for playback, but it removes the
    // element from the DOM entirely — and `document.querySelector('audio')` is how full-listen.spec
    // and the tier-3 listen-through walk observe playback. Body outlives every view, so the element
    // still survives navigation, which is the point; being inspectable costs nothing.
    audio.setAttribute('data-testid', 'app-audio')
    audio.style.display = 'none'
    if (typeof document !== 'undefined') document.body.appendChild(audio)
    audio.playbackRate = rate.value
    audio.addEventListener('play', onPlay)
    audio.addEventListener('pause', onPause)
    audio.addEventListener('timeupdate', onTimeUpdate)
    audio.addEventListener('durationchange', onDurationChange)
    audio.addEventListener('loadedmetadata', onDurationChange)
    audio.addEventListener('error', onError)
    audio.addEventListener('ended', onEnded)
    el.value = audio
    wireMediaHandlers() // headphone / BT transport works as soon as an element exists
    return audio
  }

  /**
   * Point the player at an episode. No-op when that episode is already loaded, so returning to the
   * player page mid-listen does NOT restart playback — the bug that would otherwise replace one
   * annoyance with a worse one.
   */
  function load(opts: { slug: string; url: string; title?: string | null; artwork?: string | null }): void {
    const audio = ensureElement()
    if (currentSlug.value === opts.slug && audio.src) {
      currentTitle.value = opts.title ?? currentTitle.value
      currentArtwork.value = opts.artwork ?? currentArtwork.value
      return
    }
    resetForLoad()
    currentSlug.value = opts.slug
    currentTitle.value = opts.title ?? null
    currentArtwork.value = opts.artwork ?? null
    audio.src = opts.url
    audio.playbackRate = rate.value
  }

  /** Stop and forget the current episode (sign-out, or an unplayable source). */
  function clear(): void {
    el.value?.pause()
    if (el.value) el.value.removeAttribute('src')
    currentSlug.value = null
    currentTitle.value = null
    currentArtwork.value = null
    resetForLoad()
  }

  // --- element event sinks (PlayerView's <audio> forwards these) ---
  function onPlay(): void {
    playing.value = true
    setPlaybackState('playing')
    void startBackgroundAudio() // Android foreground keep-alive (#1310); no-op on iOS/web
  }
  function onPause(): void {
    playing.value = false
    setPlaybackState('paused')
    void stopBackgroundAudio()
  }
  function onTimeUpdate(): void {
    currentTime.value = el.value?.currentTime ?? 0
    syncPositionState()
  }
  function onDurationChange(): void {
    duration.value = el.value?.duration || 0
    syncPositionState()
  }
  function onError(): void {
    audioError.value = true
    void stopBackgroundAudio()
  }
  /**
   * Advance to the next queued episode when one ends.
   *
   * PLAYS it, but does NOT navigate. Audio now outlives the player view, so an ended episode can
   * fire while the listener is reading something else entirely — yanking them to a new route would
   * be a worse interruption than the one #1587 set out to remove. The mini-player reflects the
   * change; tapping it is how you follow along.
   */
  const advanceResolvers: { next?: () => { slug: string; url: string; title?: string | null } | null } = {}
  function onEnded(): void {
    playing.value = false
    void stopBackgroundAudio()
    const next = advanceResolvers.next?.()
    if (!next) return
    load({ slug: next.slug, url: next.url, title: next.title })
    void el.value?.play()
  }
  /** The app shell supplies this; the store must not import the queue or the API itself. */
  function setAdvanceResolver(fn: (() => { slug: string; url: string; title?: string | null } | null) | undefined): void {
    advanceResolvers.next = fn
  }
  /** New episode loading — clear transient state (source swap happens in the view). */
  function resetForLoad(): void {
    playing.value = false
    currentTime.value = 0
    duration.value = 0
    audioError.value = false
    void stopBackgroundAudio()
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
    syncPositionState()
  }
  function cycleRate(): void {
    const i = PLAYBACK_RATES.indexOf(rate.value as (typeof PLAYBACK_RATES)[number])
    setRate(PLAYBACK_RATES[(i + 1) % PLAYBACK_RATES.length])
  }

  // --- MediaSession: lock-screen / notification / headphone / BT controls (#1308) ---
  // Metadata comes from the view (episode); prev/next come from the queue (view wires them). All
  // guarded so it's a no-op where MediaSession is absent (jsdom, older engines).
  const skipHandlers: { prev?: () => void; next?: () => void } = {}
  let handlersWired = false
  function hasMediaSession(): boolean {
    return typeof navigator !== 'undefined' && 'mediaSession' in navigator
  }
  function setPlaybackState(state: MediaSessionPlaybackState): void {
    if (hasMediaSession()) navigator.mediaSession.playbackState = state
  }
  function syncPositionState(): void {
    if (!hasMediaSession() || typeof navigator.mediaSession.setPositionState !== 'function') return
    const d = duration.value
    if (!d || !Number.isFinite(d)) return
    try {
      navigator.mediaSession.setPositionState({
        duration: d,
        playbackRate: rate.value || 1,
        position: Math.min(Math.max(0, currentTime.value), d),
      })
    } catch {
      /* some engines throw on out-of-range while seeking — ignore */
    }
  }
  function wireMediaHandlers(): void {
    if (handlersWired || !hasMediaSession()) return
    handlersWired = true
    const ms = navigator.mediaSession
    const set = (a: MediaSessionAction, h: MediaSessionActionHandler | null): void => {
      try {
        ms.setActionHandler(a, h)
      } catch {
        /* action unsupported on this engine — skip */
      }
    }
    set('play', () => {
      if (el.value?.paused) toggle()
    })
    set('pause', () => {
      if (el.value && !el.value.paused) toggle()
    })
    set('seekbackward', (d) => skip(-(d.seekOffset ?? 15)))
    set('seekforward', (d) => skip(d.seekOffset ?? 30))
    set('seekto', (d) => {
      if (typeof d.seekTime === 'number') seek(d.seekTime)
    })
    set('previoustrack', () => skipHandlers.prev?.())
    set('nexttrack', () => skipHandlers.next?.())
  }
  /** Lock-screen metadata for the current episode (view calls this on load). */
  function setMetadata(m: { title: string; artist?: string; album?: string; artworkUrl?: string }): void {
    if (!hasMediaSession() || typeof MediaMetadata === 'undefined') return
    navigator.mediaSession.metadata = new MediaMetadata({
      title: m.title,
      artist: m.artist ?? '',
      album: m.album ?? '',
      artwork: m.artworkUrl ? [{ src: m.artworkUrl, sizes: '512x512', type: 'image/png' }] : [],
    })
    wireMediaHandlers()
  }
  /** Prev/next track handlers (view wires them to the queue + router). */
  function setSkipHandlers(h: { prev?: () => void; next?: () => void }): void {
    skipHandlers.prev = h.prev
    skipHandlers.next = h.next
    wireMediaHandlers()
  }

  return {
    el,
    playing,
    currentTime,
    duration,
    rate,
    audioError,
    currentSlug,
    currentTitle,
    currentArtwork,
    load,
    clear,
    setAdvanceResolver,
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
    setMetadata,
    setSkipHandlers,
  }
})
