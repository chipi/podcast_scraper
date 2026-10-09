import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import type { Note } from '../services/types'
import { useAuthStore } from '../stores/auth'
import { useCaptureStore } from '../stores/capture'
import { usePlayerStore } from '../stores/player'
import NoteComposer from './NoteComposer.vue'

// useDictation captures platform capability at MODULE load, so mock the composable to drive the
// composer's mic gating / error / save-stops-dictation glue deterministically.
const dict = vi.hoisted(() => ({
  stop: vi.fn(),
  finish: vi.fn(),
  toggle: vi.fn(),
  canDictate: true,
  opts: { current: null as null | { onStart: () => void; onText: (t: string) => void; onError?: () => void } },
  dictating: { current: null as null | { value: boolean } },
}))
vi.mock('../composables/useDictation', async () => {
  const { ref } = await import('vue')
  return {
    useDictation: (opts: unknown) => {
      dict.opts.current = opts as (typeof dict)['opts']['current']
      const dictating = ref(false)
      dict.dictating.current = dictating
      return { canDictate: dict.canDictate, dictating, toggle: dict.toggle, finish: dict.finish, stop: dict.stop }
    },
  }
})
const voice = vi.hoisted(() => ({ enabled: { current: null as null | { value: boolean } } }))
vi.mock('../composables/useVoiceInput', async () => {
  const { ref } = await import('vue')
  const enabled = ref(true)
  voice.enabled.current = enabled
  return { useVoiceInput: () => ({ enabled, setEnabled: (v: boolean) => (enabled.value = v) }) }
})

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function mountComposer() {
  return mount(NoteComposer, {
    props: { target: 'episode', targetId: 'ep1' },
    global: { plugins: [i18n] },
  })
}

beforeEach(() => {
  setActivePinia(createPinia())
  useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
  vi.spyOn(api, 'getNotes').mockResolvedValue([])
  dict.stop.mockClear()
  dict.toggle.mockClear()
  dict.canDictate = true
  if (voice.enabled.current) voice.enabled.current.value = true
})

afterEach(() => vi.restoreAllMocks())

describe('NoteComposer', () => {
  it('lists notes newest first, five at a time', async () => {
    // Operator 2026-10-05: notes page in fives; the store's order is oldest-first, so without the
    // sort the newest note would be the one behind "Show more".
    const note = (i: number): Note => ({
      id: `n${i}`, target: 'episode', target_id: 'ep1', text: `note ${i}`, created_at: 1000 + i, updated_at: 1000 + i,
    } as Note)
    vi.spyOn(api, 'getHighlights').mockResolvedValue([])
    vi.spyOn(api, 'getNotes').mockResolvedValue(Array.from({ length: 7 }, (_, i) => note(i)))
    const w = mountComposer()
    await flushPromises()
    const texts = () => w.findAll('[data-testid="note-item"]').map((li) => li.find('p').text())
    expect(texts()).toEqual(['note 6', 'note 5', 'note 4', 'note 3', 'note 2'])
    await w.get('[data-testid="notes-more"]').trigger('click')
    expect(texts()).toHaveLength(7)
    expect(texts().at(-1)).toBe('note 0')
  })

  it('adds a note for the target and shows it with a timestamp', async () => {
    const created: Note = {
      id: 'n1',
      target: 'episode',
      target_id: 'ep1',
      text: 'Remember this',
      created_at: 1_700_000_000,
      updated_at: 1_700_000_000,
    }
    const create = vi.spyOn(api, 'createNote').mockResolvedValue(created)
    const w = mountComposer()

    await w.get('[data-testid="note-input"]').setValue('Remember this')
    await w.get('[data-testid="note-save"]').trigger('click')
    await flushPromises()

    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ target: 'episode', target_id: 'ep1', text: 'Remember this' }),
    )
    const item = w.get('[data-testid="note-item"]')
    expect(item.text()).toContain('Remember this')
    // A timestamp (NT.2) renders alongside the note.
    expect(item.find('.lp-kicker').exists()).toBe(true)
  })

  it('deleting a note ASKS first: Cancel keeps it, confirming deletes it (UXS-014)', async () => {
    const note: Note = {
      id: 'n1', target: 'episode', target_id: 'ep1', text: 'Keep me?',
      created_at: 1_700_000_000, updated_at: 1_700_000_000,
    }
    vi.spyOn(api, 'getNotes').mockResolvedValue([note])
    vi.spyOn(api, 'getHighlights').mockResolvedValue([])
    const del = vi.spyOn(api, 'deleteNote').mockResolvedValue([])
    const w = mountComposer()
    await flushPromises()

    await w.get('[data-testid="note-delete"]').trigger('click')
    await flushPromises()
    expect(del, 'one tap deleted the note without asking').not.toHaveBeenCalled()
    await w.get('[data-testid="confirm-cancel"]').trigger('click')
    await flushPromises()
    expect(del).not.toHaveBeenCalled()
    expect(w.text()).toContain('Keep me?')

    await w.get('[data-testid="note-delete"]').trigger('click')
    await flushPromises()
    await w.get('[data-testid="confirm-accept"]').trigger('click')
    await flushPromises()
    expect(del).toHaveBeenCalledWith('n1')
  })

  it('does not save an empty note', async () => {
    const create = vi.spyOn(api, 'createNote')
    const w = mountComposer()
    // Save is disabled with an empty draft.
    expect(w.get('[data-testid="note-save"]').attributes('disabled')).toBeDefined()
    await w.get('[data-testid="note-save"]').trigger('click')
    expect(create).not.toHaveBeenCalled()
  })

  it('shows the mic only when voice input is ON and the platform can dictate', async () => {
    const w = mountComposer()
    await flushPromises()
    expect(w.find('[data-testid="note-dictate"]').exists()).toBe(true)
    // Turning the Settings opt-in off hides it (reactive), even though the platform can dictate.
    voice.enabled.current!.value = false
    await flushPromises()
    expect(w.find('[data-testid="note-dictate"]').exists()).toBe(false)
  })

  it('surfaces a dictation failure via the onError hook', async () => {
    const w = mountComposer()
    await flushPromises()
    expect(w.find('[data-testid="note-dictate-error"]').exists()).toBe(false)
    dict.opts.current!.onError!()
    await flushPromises()
    expect(w.find('[data-testid="note-dictate-error"]').exists()).toBe(true)
  })

  it('the mic tap ends dictation with finish(), which keeps the last word; Save uses stop()', async () => {
    const w = mountComposer()
    await flushPromises()
    dict.dictating.current!.value = true
    await flushPromises()
    await w.get('[data-testid="note-dictate"]').trigger('click')
    expect(dict.finish).toHaveBeenCalledTimes(1)
    expect(dict.stop).not.toHaveBeenCalled()
  })

  it('stops an active dictation before saving so a late partial cannot resurrect the draft', async () => {
    vi.spyOn(api, 'createNote').mockResolvedValue({
      id: 'n1',
      target: 'episode',
      target_id: 'ep1',
      text: 'note text',
      created_at: 1,
      updated_at: 1,
    })
    const w = mountComposer()
    await w.get('[data-testid="note-input"]').setValue('note text')
    await w.get('[data-testid="note-save"]').trigger('click')
    await flushPromises()
    expect(dict.stop).toHaveBeenCalled()
  })

  describe('an episode the mic paused (Android audio focus, 2026-10-09)', () => {
    /** The mic starts while the episode plays; `pauseAfterMs` later, something pauses it. */
    async function dictateOver(opts: { playing: boolean; pauseAfterMs: number }) {
      vi.useFakeTimers()
      const player = usePlayerStore()
      const resume = vi.spyOn(player, 'resumeAfterInterruption').mockImplementation(() => {})
      player.playing = opts.playing
      mountComposer()
      await flushPromises()
      dict.opts.current!.onStart()
      dict.dictating.current!.value = true
      await flushPromises()
      vi.advanceTimersByTime(opts.pauseAfterMs)
      player.playing = false
      await flushPromises()
      dict.dictating.current!.value = false
      await flushPromises()
      vi.useRealTimers()
      return resume
    }

    it('resumes it when dictation ends', async () => {
      // Measured on the emulator: the recogniser took audio focus, the WebView paused the episode,
      // and nothing ever started it again.
      expect(await dictateOver({ playing: true, pauseAfterMs: 200 })).toHaveBeenCalledTimes(1)
    })

    it('leaves alone a pause the reader made later, mid-dictation', async () => {
      expect(await dictateOver({ playing: true, pauseAfterMs: 8000 })).not.toHaveBeenCalled()
    })

    it('does not start an episode that was not playing', async () => {
      expect(await dictateOver({ playing: false, pauseAfterMs: 200 })).not.toHaveBeenCalled()
    })

    it('resumes it when the mic fails to start after taking the audio', async () => {
      vi.useFakeTimers()
      const player = usePlayerStore()
      const resume = vi.spyOn(player, 'resumeAfterInterruption').mockImplementation(() => {})
      player.playing = true
      mountComposer()
      await flushPromises()
      dict.opts.current!.onStart()
      player.playing = false
      await flushPromises()
      dict.opts.current!.onError!()
      await flushPromises()
      vi.useRealTimers()
      expect(resume).toHaveBeenCalledTimes(1)
    })
  })
})
