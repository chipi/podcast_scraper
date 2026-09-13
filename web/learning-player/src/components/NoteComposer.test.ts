import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import type { Note } from '../services/types'
import { useAuthStore } from '../stores/auth'
import { useCaptureStore } from '../stores/capture'
import NoteComposer from './NoteComposer.vue'

// useDictation captures platform capability at MODULE load, so mock the composable to drive the
// composer's mic gating / error / save-stops-dictation glue deterministically.
const dict = vi.hoisted(() => ({
  stop: vi.fn(),
  toggle: vi.fn(),
  canDictate: true,
  opts: { current: null as null | { onStart: () => void; onText: (t: string) => void; onError?: () => void } },
}))
vi.mock('../composables/useDictation', async () => {
  const { ref } = await import('vue')
  return {
    useDictation: (opts: unknown) => {
      dict.opts.current = opts as (typeof dict)['opts']['current']
      return { canDictate: dict.canDictate, dictating: ref(false), toggle: dict.toggle, stop: dict.stop }
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
})
