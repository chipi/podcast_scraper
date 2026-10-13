/**
 * Profile › Interests › Suggested less (ADR-163 §6): lists what Discover shows less of, a tap
 * undoes it, and it is absent when there is nothing to list.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createPinia } from 'pinia'
import en from '../i18n/locales/en.json'

const api = vi.hoisted(() => ({
  getSuggestLess: vi.fn(),
  undoSuggestLess: vi.fn(),
  getGoneReferences: vi.fn().mockResolvedValue([]),
}))
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...api }))

import SuggestedLessList from './SuggestedLessList.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const row = (token: string, label: string) => ({ token, label, added_at: 1 })

describe('SuggestedLessList', () => {
  it('lists what was hidden; a tap undoes it', async () => {
    api.getSuggestLess.mockResolvedValue([row('topic:a', 'Alpha'), row('person:b', 'Bea')])
    api.undoSuggestLess.mockResolvedValue([row('person:b', 'Bea')])
    const w = mount(SuggestedLessList, { global: { plugins: [i18n, createPinia()] } })
    await flushPromises()
    const items = w.findAll('[data-testid="suggested-less-item"]')
    expect(items.map((i) => i.text())).toEqual(['Alpha ×', 'Bea ×'])
    await items[0].trigger('click')
    await flushPromises()
    expect(api.undoSuggestLess).toHaveBeenCalledWith('topic:a')
    expect(w.findAll('[data-testid="suggested-less-item"]').map((i) => i.text())).toEqual(['Bea ×'])
  })

  it('is absent with nothing hidden, and says so when it cannot load', async () => {
    api.getSuggestLess.mockResolvedValue([])
    const empty = mount(SuggestedLessList, { global: { plugins: [i18n, createPinia()] } })
    await flushPromises()
    expect(empty.find('[data-testid="suggested-less"]').exists()).toBe(false)
    api.getSuggestLess.mockRejectedValue(new Error('offline'))
    const failed = mount(SuggestedLessList, { global: { plugins: [i18n, createPinia()] } })
    await flushPromises()
    expect(failed.find('[data-testid="suggested-less-unavailable"]').exists()).toBe(true)
  })

  it('a subject the corpus no longer has is shown muted, "No longer available", and still undone', async () => {
    api.getGoneReferences.mockResolvedValueOnce([{ ref: 'thc:gone', kind: 'storyline' }])
    api.getSuggestLess.mockResolvedValue([row('thc:gone', 'Old thread'), row('topic:a', 'Alpha')])
    const w = mount(SuggestedLessList, { global: { plugins: [i18n, createPinia()] } })
    await flushPromises()
    const [gone, ok] = w.findAll('[data-testid="suggested-less-item"]')
    expect(gone.text()).toContain('No longer available')
    expect(gone.classes()).toContain('opacity-60')
    expect(ok.text()).not.toContain('No longer available')
  })
})
