import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import KeyVoicesRail from './KeyVoicesRail.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/person/:id', name: 'person', component: { template: '<div/>' } },
  ],
})

function mountRail() {
  return mount(KeyVoicesRail, { global: { plugins: [i18n, router] } })
}

describe('KeyVoicesRail (wave-G)', () => {
  beforeEach(() => vi.restoreAllMocks())

  it('renders a chip per voice, linking to the person card', async () => {
    vi.spyOn(api, 'getKeyVoices').mockResolvedValue({
      voices: [
        { id: 'person:jane', kind: 'person', label: 'Jane Doe', episode_count: 5 },
        { id: 'person:john', kind: 'person', label: 'John Roe', episode_count: 3 },
      ],
    })
    const w = mountRail()
    await flushPromises()
    expect(w.find('[data-testid="key-voices-rail"]').exists()).toBe(true)
    const chips = w.findAll('[data-testid="key-voice"]')
    expect(chips).toHaveLength(2)
    expect(chips[0].attributes('href')).toBe('/person/person:jane')
    expect(w.text()).toContain('Jane Doe')
  })

  it('self-hides when there are no voices', async () => {
    vi.spyOn(api, 'getKeyVoices').mockResolvedValue({ voices: [] })
    const w = mountRail()
    await flushPromises()
    expect(w.find('[data-testid="key-voices-rail"]').exists()).toBe(false)
  })

  it('self-hides (never throws) when the fetch fails', async () => {
    vi.spyOn(api, 'getKeyVoices').mockRejectedValue(new Error('offline'))
    const w = mountRail()
    await flushPromises()
    expect(w.find('[data-testid="key-voices-rail"]').exists()).toBe(false)
  })
})
