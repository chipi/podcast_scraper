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
})
