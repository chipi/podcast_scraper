import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { effectScope } from 'vue'
import type { Dictation, DictationOptions } from './useDictation'

// `isNative` and the Web Speech constructor are captured at MODULE LOAD, so each case configures the
// platform, then `vi.resetModules()` + a fresh dynamic import re-evaluates those constants.
const isNativePlatform = vi.fn(() => false)
vi.mock('@capacitor/core', () => ({
  Capacitor: { isNativePlatform: () => isNativePlatform(), getPlatform: () => 'web' },
}))

type Listener = (data: unknown) => void
const nativeListeners: Record<string, Listener> = {}
const removedEvents: string[] = []
const native = {
  available: vi.fn(async () => ({ available: true })),
  requestPermissions: vi.fn(async () => ({ speechRecognition: 'granted', microphone: 'granted' })),
  addListener: vi.fn(async (event: string, cb: Listener) => {
    nativeListeners[event] = cb
    return {
      remove: vi.fn(async () => {
        removedEvents.push(event)
        delete nativeListeners[event]
      }),
    }
  }),
  start: vi.fn(async () => {}),
  stop: vi.fn(async () => {}),
  removeAllListeners: vi.fn(async () => {}),
}
vi.mock('@capacitor-community/speech-recognition', () => ({ SpeechRecognition: native }))

class FakeWebSR {
  static instances: FakeWebSR[] = []
  interimResults = false
  lang = ''
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null = null
  onend: (() => void) | null = null
  onerror: (() => void) | null = null
  start = vi.fn()
  stop = vi.fn()
  constructor() {
    FakeWebSR.instances.push(this)
  }
}

const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0))

/** A caller harness that mirrors NoteComposer: a draft the transcript appends to, and error count. */
function harness(): { draft: { value: string }; errors: number; opts: DictationOptions } {
  const draft = { value: '' }
  let base = ''
  const h = {
    draft,
    errors: 0,
    opts: {
      lang: () => 'en',
      onStart: () => {
        base = draft.value ? draft.value + ' ' : ''
      },
      onText: (t: string) => {
        draft.value = base + t
      },
      onError: () => {
        h.errors += 1
      },
    },
  }
  return h
}

async function load(opts: {
  native: boolean
  web: boolean
}): Promise<(o: DictationOptions) => Dictation> {
  isNativePlatform.mockReturnValue(opts.native)
  const w = window as unknown as { SpeechRecognition?: unknown; webkitSpeechRecognition?: unknown }
  if (opts.web) w.SpeechRecognition = FakeWebSR
  else {
    delete w.SpeechRecognition
    delete w.webkitSpeechRecognition
  }
  vi.resetModules()
  return (await import('./useDictation')).useDictation
}

/** Run `useDictation` inside an effect scope so `onScopeDispose(stop)` is exercised by `scope.stop()`. */
function inScope(useDictation: (o: DictationOptions) => Dictation, opts: DictationOptions) {
  const scope = effectScope()
  let d!: Dictation
  scope.run(() => {
    d = useDictation(opts)
  })
  return { d, dispose: () => scope.stop() }
}

beforeEach(() => {
  FakeWebSR.instances = []
  for (const k of Object.keys(nativeListeners)) delete nativeListeners[k]
  removedEvents.length = 0
  native.available.mockResolvedValue({ available: true })
  native.requestPermissions.mockResolvedValue({ speechRecognition: 'granted', microphone: 'granted' })
})
afterEach(() => vi.clearAllMocks())

describe('useDictation — web engine', () => {
  it('appends the transcript to the existing draft rather than replacing it', async () => {
    const useDictation = await load({ native: false, web: true })
    const h = harness()
    h.draft.value = 'Existing.'
    const { d } = inScope(useDictation, h.opts)

    d.toggle()
    expect(d.dictating.value).toBe(true)
    const inst = FakeWebSR.instances.at(-1)!
    inst.onresult!({ results: [[{ transcript: 'new words' }]] })
    expect(h.draft.value).toBe('Existing. new words')
  })

  it('surfaces a permission/engine error and drops the "on" latch (M1)', async () => {
    const useDictation = await load({ native: false, web: true })
    const h = harness()
    const { d } = inScope(useDictation, h.opts)

    d.toggle()
    const inst = FakeWebSR.instances.at(-1)!
    inst.onerror!()
    expect(d.dictating.value).toBe(false)
    expect(h.errors).toBe(1)
  })

  it('a superseded instance\'s late onend cannot clear the new session (M2)', async () => {
    const useDictation = await load({ native: false, web: true })
    const h = harness()
    const { d } = inScope(useDictation, h.opts)

    d.toggle()
    const first = FakeWebSR.instances.at(-1)!
    d.stop()
    d.toggle()
    expect(d.dictating.value).toBe(true)
    // The first recogniser's `onend` fires late — it must not turn off the live second session.
    first.onend!()
    expect(d.dictating.value).toBe(true)
  })

  it('a late result delivered after stop() does not resurrect the cleared draft (H1)', async () => {
    const useDictation = await load({ native: false, web: true })
    const h = harness()
    h.draft.value = 'typed'
    const { d } = inScope(useDictation, h.opts)

    d.toggle()
    const inst = FakeWebSR.instances.at(-1)!
    d.stop()
    h.draft.value = '' // the caller cleared the draft (e.g. saved the note)
    // Chrome fires a final `result` after `.stop()` when audio is pending — it must be ignored.
    inst.onresult!({ results: [[{ transcript: 'late words' }]] })
    expect(h.draft.value).toBe('')
  })

  it('reports an error when no engine exists at all', async () => {
    const useDictation = await load({ native: false, web: false })
    const h = harness()
    const { d } = inScope(useDictation, h.opts)
    expect(d.canDictate).toBe(false)
    d.toggle()
    expect(h.errors).toBe(1)
    expect(d.dictating.value).toBe(false)
  })
})

describe('useDictation — native engine', () => {
  it('does not latch "on" when permission is denied, and reports it', async () => {
    native.requestPermissions.mockResolvedValue({ speechRecognition: 'denied', microphone: 'denied' })
    const useDictation = await load({ native: true, web: false })
    const h = harness()
    const { d } = inScope(useDictation, h.opts)

    d.toggle()
    await flush()
    expect(native.start).not.toHaveBeenCalled()
    expect(d.dictating.value).toBe(false)
    expect(h.errors).toBe(1)
  })

  it('blocks Android when the mic grant is missing even though speech is granted', async () => {
    native.requestPermissions.mockResolvedValue({ speechRecognition: 'granted', microphone: 'denied' })
    const useDictation = await load({ native: true, web: false })
    const h = harness()
    const { d } = inScope(useDictation, h.opts)

    d.toggle()
    await flush()
    expect(native.start).not.toHaveBeenCalled()
    expect(d.dictating.value).toBe(false)
    expect(h.errors).toBe(1)
  })

  it('a second tap during the async start window does not open a second recogniser (H1 latch)', async () => {
    const useDictation = await load({ native: true, web: false })
    const h = harness()
    const { d } = inScope(useDictation, h.opts)

    d.toggle()
    d.toggle() // still resolving permission — must be ignored
    await flush()
    expect(native.start).toHaveBeenCalledTimes(1)
    expect(d.dictating.value).toBe(true)
  })

  it('stop() during the start window aborts — the mic never turns on afterwards (H1 token)', async () => {
    const useDictation = await load({ native: true, web: false })
    const h = harness()
    const { d } = inScope(useDictation, h.opts)

    d.toggle()
    d.stop() // navigates away mid-permission-prompt
    await flush()
    expect(native.start).not.toHaveBeenCalled()
    expect(d.dictating.value).toBe(false)
  })

  it('a self-stop (silence timeout) drops the "on" state and detaches its listeners (M3)', async () => {
    const useDictation = await load({ native: true, web: false })
    const h = harness()
    const { d } = inScope(useDictation, h.opts)

    d.toggle()
    await flush()
    expect(d.dictating.value).toBe(true)
    nativeListeners['listeningState']!({ status: 'stopped' })
    await flush()
    expect(d.dictating.value).toBe(false)
    expect(removedEvents).toContain('partialResults')
    expect(removedEvents).toContain('listeningState')
  })

  it('a native partial bridged in after stop() does not rewrite the draft (M1)', async () => {
    const useDictation = await load({ native: true, web: false })
    const h = harness()
    const { d } = inScope(useDictation, h.opts)

    d.toggle()
    await flush()
    const partial = nativeListeners['partialResults']!
    d.stop()
    h.draft.value = ''
    partial({ matches: ['late'] })
    expect(h.draft.value).toBe('')
  })

  it('disposing the scope stops the engine (onScopeDispose)', async () => {
    const useDictation = await load({ native: true, web: false })
    const h = harness()
    const { d, dispose } = inScope(useDictation, h.opts)
    d.toggle()
    await flush()
    dispose()
    expect(native.stop).toHaveBeenCalled()
    expect(d.dictating.value).toBe(false)
  })
})
