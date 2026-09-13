import { onScopeDispose, ref, type Ref } from 'vue'
import { Capacitor } from '@capacitor/core'
import { SpeechRecognition } from '@capacitor-community/speech-recognition'

/**
 * Note dictation, one interface over two engines (operator 2026-09-13):
 *
 * - **Native (iOS/Android)** — the `@capacitor-community/speech-recognition` plugin. The browser
 *   Web Speech API exists in the iOS WKWebView but never fires a result, so the mic latched yellow
 *   and transcribed nothing; the plugin is the only thing that actually records there.
 * - **Browser** — the built-in `SpeechRecognition` / `webkitSpeechRecognition`, unchanged.
 *
 * The caller owns the draft: `onStart` snapshots the base text, `onText` receives the running
 * transcript. `dictating` latches true ONLY once recording actually begins — on native that means
 * after availability + permission both pass, so a denied mic no longer looks "on".
 */
interface SpeechRecognitionLike {
  interimResults: boolean
  lang: string
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null
  onend: (() => void) | null
  onerror: (() => void) | null
  start(): void
  stop(): void
}
type SRCtor = new () => SpeechRecognitionLike

const WebSR: SRCtor | undefined =
  typeof window === 'undefined'
    ? undefined
    : (window as unknown as { SpeechRecognition?: SRCtor; webkitSpeechRecognition?: SRCtor })
        .SpeechRecognition ??
      (window as unknown as { webkitSpeechRecognition?: SRCtor }).webkitSpeechRecognition

const isNative = Capacitor.isNativePlatform()

export interface DictationOptions {
  /** BCP-47 language for the recogniser, read lazily so a locale change is picked up. */
  lang: () => string
  /** Snapshot the current draft so appended text is added to it, not replacing it. */
  onStart: () => void
  /** The running transcript (interim on web, partial on native). */
  onText: (text: string) => void
  /**
   * Dictation could not start OR failed while running — unavailable engine, denied permission (web
   * or native), or an engine error. Without this the failure is silent and reads as a dead mic.
   */
  onError?: () => void
}

export interface Dictation {
  /** The PLATFORM can dictate. The caller still gates on the Settings opt-in. */
  canDictate: boolean
  dictating: Ref<boolean>
  toggle: () => void
  stop: () => void
}

/** A subscription we own and must detach individually — the plugin's `removeAllListeners()` is
 *  global and would also tear down any sibling component's listeners. */
type ListenerHandle = { remove: () => Promise<void> }

export function useDictation(opts: DictationOptions): Dictation {
  const dictating = ref(false)
  let webRecog: SpeechRecognitionLike | null = null
  // A native start() spans async availability + permission (a permission prompt can take seconds).
  // Without this latch a second tap in that window would open a second recogniser fighting the mic.
  let starting = false
  // Our own listener handles, so stop() detaches only what this composable added.
  let partialHandle: ListenerHandle | null = null
  let stateHandle: ListenerHandle | null = null
  // Generation token, bumped by every stop()/dispose. A native start still resolving its async
  // prompt compares against it after each await and ABORTS if it changed — otherwise stopping (or
  // navigating away) mid-prompt would turn the mic on afterwards, on a dead scope.
  let session = 0

  // Native: the plugin is installed, so the platform is capable in principle; runtime availability
  // + permission are resolved at start(). Browser: needs the Web Speech constructor.
  const canDictate = isNative || !!WebSR

  async function detachNative(): Promise<void> {
    // Null the refs SYNCHRONOUSLY, then await removal on the captured handles, so a start that
    // races in between attaches fresh handles this call does not clobber.
    const p = partialHandle
    const s = stateHandle
    partialHandle = null
    stateHandle = null
    await p?.remove().catch(() => {})
    await s?.remove().catch(() => {})
  }

  async function startNative(): Promise<void> {
    const mySession = session
    const aborted = (): boolean => mySession !== session
    const { available } = await SpeechRecognition.available().catch(() => ({ available: false }))
    if (aborted()) return
    if (!available) {
      opts.onError?.()
      return
    }
    const perm = await SpeechRecognition.requestPermissions().catch(() => null)
    if (aborted()) return
    // Android prompts for speech recognition AND the microphone as two separate grants; a
    // granted-speech / denied-mic start() fails at the OS layer AFTER we would have latched "on".
    // The plugin's typed status only declares `speechRecognition`; Android also returns `microphone`
    // at runtime, so read it defensively — an absent field (iOS) must not fail the gate.
    const micState = (perm as { microphone?: string } | null)?.microphone
    const granted =
      perm?.speechRecognition === 'granted' && (micState === undefined || micState === 'granted')
    if (!granted) {
      opts.onError?.()
      return
    }
    // Permission is confirmed — snapshot the draft only now, so text the user typed during the
    // prompt is not captured into a stale base and then lost.
    opts.onStart()
    await detachNative()
    if (aborted()) return
    partialHandle = await SpeechRecognition.addListener(
      'partialResults',
      (data: { matches: string[] }) => {
        // A partial bridged in after this session was superseded/stopped must not rewrite the draft.
        if (aborted()) return
        if (data.matches?.length) opts.onText(data.matches[0])
      },
    )
    stateHandle = await SpeechRecognition.addListener(
      'listeningState',
      (data: { status: string }) => {
        // Ignore a stale engine's stop event landing on a newer session's listeners.
        if (aborted()) return
        if (data.status === 'stopped') {
          // The engine self-stopped (e.g. an iOS silence timeout) — drop OUR listeners too, or a
          // late partial event would keep rewriting the draft after the mic reads "off".
          dictating.value = false
          void detachNative()
        }
      },
    )
    if (aborted()) {
      await detachNative()
      return
    }
    try {
      await SpeechRecognition.start({
        language: opts.lang() || 'en-US',
        partialResults: true,
        popup: false,
      })
      if (aborted()) {
        // Stopped during the final await — actively turn the engine back off, don't just bail.
        await SpeechRecognition.stop().catch(() => {})
        await detachNative()
        return
      }
      dictating.value = true
    } catch {
      // The engine refused after listeners were attached — tear them back down and stay "off".
      await detachNative()
      // A rejection CAUSED by an intentional stop() during this await is not a user-facing error.
      if (aborted()) return
      dictating.value = false
      opts.onError?.()
    }
  }

  function startWeb(): void {
    if (!WebSR) {
      opts.onError?.()
      return
    }
    opts.onStart()
    // Capture the instance so a superseded recogniser (from a fast stop→start) cannot write the new
    // session's state through its own late `onend` / `onresult` / `onerror`.
    const inst = new WebSR()
    webRecog = inst
    inst.interimResults = true
    inst.lang = opts.lang() || 'en'
    inst.onresult = (e) => {
      if (webRecog !== inst) return
      let transcript = ''
      for (let i = 0; i < e.results.length; i++) transcript += e.results[i][0].transcript
      opts.onText(transcript)
    }
    inst.onend = () => {
      if (webRecog === inst) dictating.value = false
    }
    inst.onerror = () => {
      // A denied mic / no-speech on web fires error then end; without this the mic latched on and
      // then silently flipped off with no signal to the caller.
      if (webRecog === inst) {
        dictating.value = false
        opts.onError?.()
      }
    }
    try {
      inst.start()
    } catch {
      // `start()` throws InvalidStateError if a recogniser is already running — don't latch "on".
      if (webRecog === inst) webRecog = null
      opts.onError?.()
      return
    }
    dictating.value = true
  }

  function toggle(): void {
    if (dictating.value) {
      stop()
      return
    }
    // A start is already resolving its permission prompt — ignore the double tap rather than open a
    // competing recogniser.
    if (starting) return
    if (isNative) {
      starting = true
      void startNative().finally(() => {
        starting = false
      })
    } else {
      startWeb()
    }
  }

  function stop(): void {
    // Abort any native start still resolving its async prompt (see `session`), and free the latch so
    // a fresh start can begin immediately (the aborted start won't actually engage the engine).
    session++
    starting = false
    if (isNative) {
      void SpeechRecognition.stop().catch(() => {})
      void detachNative()
    } else {
      // Null the ref BEFORE stopping so the instance's own late `onresult`/`onend`/`onerror` (Chrome
      // delivers a final result after stop()) fail the `webRecog === inst` guard and can't rewrite
      // the just-cleared draft.
      const inst = webRecog
      webRecog = null
      inst?.stop()
    }
    dictating.value = false
  }

  onScopeDispose(() => stop())
  return { canDictate, dictating, toggle, stop }
}
