import { defineStore } from 'pinia'
import { ref } from 'vue'
import { PLAYBACK_RATES } from '../player/transcriptSync'
import { startBackgroundAudio, stopBackgroundAudio } from '../services/native'

/**
 * What it takes to start playing something: the identity, the source, and enough to say what it is.
 *
 * Title and artwork are optional in the type but effectively required in practice — every surface
 * that shows "now playing" (mini-player, lock screen, headphones, car) reads them from the store,
 * and auto-advance runs with no view mounted to supply them later.
 */
export interface NextUp {
  slug: string
  url: string
  title?: string | null
  artwork?: string | null
}

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
    // Closing the tab is not an unmount: onBeforeUnmount fires on SPA navigation only, so a tab
    // closed mid-episode dropped everything since the last throttled save. `pagehide` is the event
    // that actually covers tab close, back/forward cache and mobile app-switch.
    if (typeof window !== 'undefined') window.addEventListener('pagehide', () => savePosition())
    return audio
  }

  /**
   * Point the player at an episode. No-op when that episode is already loaded, so returning to the
   * player page mid-listen does NOT restart playback — the bug that would otherwise replace one
   * annoyance with a worse one.
   */
  function load(opts: NextUp): void {
    const audio = ensureElement()
    if (currentSlug.value === opts.slug && audio.src) {
      currentTitle.value = opts.title ?? currentTitle.value
      currentArtwork.value = opts.artwork ?? currentArtwork.value
      return
    }
    // Flush the OUTGOING episode before its identity is overwritten — otherwise up to a full
    // throttle window of the episode the user just left is lost, every single switch.
    savePosition()
    resetForLoad()
    currentSlug.value = opts.slug
    currentTitle.value = opts.title ?? null
    currentArtwork.value = opts.artwork ?? null
    // A locally downloaded copy wins over the origin URL (#1905). The resolver is INJECTED by the
    // shell, exactly like the advance resolver: this store must not import the downloads store or
    // the API, or playback stops being independent of data fetching (stores/README.md).
    // Sync on purpose — load() is sync and every caller depends on that.
    audio.src = sourceResolvers.local?.(opts.slug) ?? opts.url
    audio.playbackRate = rate.value
    // Every path into load() owns the now-playing identity, including auto-advance, which happens
    // with no view mounted. Without this the lock screen, headphones and car display keep showing
    // the PREVIOUS episode for the whole of the next one. PlayerView still calls setMetadata with
    // richer data (artist/album) when it is mounted; this is the floor, not the ceiling.
    if (opts.title) {
      setMetadata({ title: opts.title, artworkUrl: opts.artwork ?? undefined })
    }
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
    // Pausing is the clearest "I am done for now" signal there is; not flushing here meant losing
    // up to the whole throttle window at the exact moment the position matters most.
    savePosition()
  }
  function onTimeUpdate(): void {
    currentTime.value = el.value?.currentTime ?? 0
    syncPositionState()
    maybeSavePosition()
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
  const advanceResolvers: { next?: () => Promise<NextUp | null> } = {}
  /** Slug -> playable local file src, or null to stream from the origin. Injected by the shell. */
  const sourceResolvers: { local?: (slug: string) => string | null } = {}
  /**
   * The resolver is ASYNC and is called at `ended`, not at load.
   *
   * An earlier version resolved the next episode when the current one *started* and cached it. That
   * froze the answer for the whole episode: "Play next", a reorder, or anything else the user did
   * while listening — which is when every such input happens — was ignored, and an empty queue at
   * start meant silence at the end even after the user queued something. It also meant the audio
   * URL was fetched an hour or two before use, so a signed origin URL could expire before playback.
   *
   * Resolving on demand costs a short gap between tracks, which every streaming player has.
   */
  async function onEnded(): Promise<void> {
    playing.value = false
    void stopBackgroundAudio()
    // Record the finish BEFORE load() overwrites currentSlug — otherwise the episode that just
    // ended keeps its last cadence save, parked seconds from the end, and stays in Continue.
    savePosition(true)
    const next = await advanceResolvers.next?.()
    if (!next) return
    load(next)
    play()
  }

  /**
   * Start playback, recording a refusal instead of dropping it.
   *
   * `el.play()` returns a promise that REJECTS on a dead source, and both call sites used to
   * `void` it. Auto-advancing into a broken episode therefore produced silence with a mini-player
   * showing the new title, paused, and no explanation anywhere; pressing play on a dead source did
   * nothing, repeatably, with no feedback. The element's `error` event does not reliably cover
   * this — a rejected play() is its own signal.
   *
   * NotAllowedError is excluded: that is the browser's autoplay policy asking for a gesture, not a
   * broken episode, and flagging it would tell the user their audio is unavailable when it is fine.
   */
  function play(): void {
    el.value?.play().catch((err: unknown) => {
      if (err instanceof DOMException && err.name === 'NotAllowedError') return
      audioError.value = true
      void stopBackgroundAudio()
    })
  }
  /** The app shell supplies this; the store must not import the queue or the API itself. */
  function setAdvanceResolver(fn: (() => Promise<NextUp | null>) | undefined): void {
    advanceResolvers.next = fn
  }

  /** Injected by the shell so downloaded episodes play from disk instead of the network. */
  function setSourceResolver(fn: ((slug: string) => string | null) | undefined): void {
    sourceResolvers.local = fn
  }

  // --- playback position persistence ------------------------------------------------------------
  //
  // This belongs to the store because the invariant is about the store's own state: a position
  // write must take the slug and the time from the SAME object. PlayerView used to own it and paired
  // `props.slug` (the page) with the store's `currentTime` (whatever is actually playing) — two
  // things that stopped being the same the moment the element outlived the view. Two ways they
  // diverged, both silent:
  //
  //   * Auto-advance while sitting on the player page. Episode X ends on /player/X, `onEnded` loads
  //     and plays queue-next Y WITHOUT navigating (deliberately — see onEnded). The view stayed
  //     mounted with props.slug === X, so every position save for Y was written onto X's record,
  //     every 10s, for the whole of Y. Opening X later "resumed" at a point in a different episode.
  //   * Switching episodes. props.slug changes immediately on tap while the old episode keeps
  //     playing through the awaits before `load()` swaps the source; any save in that window put the
  //     old episode's time on the new slug.
  //
  // Moving it here also fixes the other half: persistence used to exist ONLY while PlayerView was
  // mounted, so listening via the mini-player — the entire point of #1587 — saved nothing at all.
  // An hour of listening from Home left the resume point where the player page had last seen it.
  //
  // The store still must not import the API (same rule as the advance resolver), so the shell
  // supplies the writer.
  const SAVE_INTERVAL_MS = 10_000
  /**
   * Past this fraction of the episode, it counts as finished.
   *
   * `ended` alone is not enough: skipping the last minute of outro is a normal way to finish an
   * episode, and `ended` never fires for it. Without the threshold those episodes sit in "Continue
   * listening" forever — which is exactly the state the flag exists to prevent.
   */
  const FINISHED_FRACTION = 0.95
  const persisters: {
    save?: (slug: string, seconds: number, finished: boolean) => void
  } = {}
  let lastSavedAt = 0

  /** The shell supplies this; without it every save below is a no-op (tests, signed-out). */
  function setPositionPersister(
    fn: ((slug: string, seconds: number, finished: boolean) => void) | undefined,
  ): void {
    persisters.save = fn
    lastSavedAt = 0
  }

  /** Write the CURRENT episode's position now. Slug and time are read together, on purpose.
   *
   * Time comes off the ELEMENT rather than the mirrored ref: the ref only moves on `timeupdate`
   * (~4/s), so a flush triggered by pause or pagehide would otherwise write a slightly stale
   * position — at the two moments the position matters most.
   */
  function savePosition(finished = false): void {
    const slug = currentSlug.value
    if (!slug || !persisters.save) return
    lastSavedAt = Date.now()
    const at = el.value?.currentTime ?? currentTime.value
    const d = duration.value
    persisters.save(slug, at, finished || (d > 0 && at / d >= FINISHED_FRACTION))
  }

  /** Throttled save for the `timeupdate` firehose (~4/s). */
  function maybeSavePosition(): void {
    if (Date.now() - lastSavedAt > SAVE_INTERVAL_MS) savePosition()
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
    if (e.paused) play()
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
    setSourceResolver,
    setPositionPersister,
    savePosition,
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
