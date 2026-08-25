import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { Collection, CollectionDetail } from '../services/types'
import CollectionsView from './CollectionsView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/player/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

function col(over: Partial<Collection> = {}): Collection {
  return { id: 'col_1', name: 'AI takes', created_at: 1, count: 2, ...over }
}

const mountView = () => mount(CollectionsView, { global: { plugins: [i18n, router] } })

afterEach(() => vi.restoreAllMocks())

describe('CollectionsView', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
  })

  it('lists collections with their counts', async () => {
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('AI takes')
    expect(w.text()).toContain('2 items')
  })

  it('creates a collection and prepends it', async () => {
    const create = vi
      .spyOn(api, 'createCollection')
      .mockResolvedValue(col({ id: 'col_2', name: 'ML', count: 0 }))
    const w = mountView()
    await flushPromises()
    await w.find('input[type="text"]').setValue('ML')
    await w.find('form').trigger('submit')
    await flushPromises()
    expect(create).toHaveBeenCalledWith('ML')
    expect(w.text()).toContain('ML')
  })

  it('opens a collection and renders its mixed items, and removes one', async () => {
    const detail: CollectionDetail = {
      collection: col(),
      items: [
        { kind: 'highlight', ref: 'h1', title: 'a line', deep_link: '/player/ep' },
        { kind: 'episode', ref: 'ep-x', title: 'An episode', deep_link: '/episode/ep-x' },
        { kind: 'link', ref: 'https://ex.com/p', title: 'A post', deep_link: 'https://ex.com/p' },
      ],
    }
    vi.spyOn(api, 'getCollection').mockResolvedValue(detail)
    const remove = vi.spyOn(api, 'removeFromCollection').mockResolvedValue(col({ count: 2 }))
    const w = mountView()
    await flushPromises()
    await w.findAll('button').find((b) => b.text().includes('AI takes'))!.trigger('click')
    await flushPromises()
    expect(w.text()).toContain('a line')
    expect(w.text()).toContain('An episode')
    // Link opens externally; in-app items use RouterLink.
    expect(w.find('a[href="https://ex.com/p"]').exists()).toBe(true)
    // Remove the first item → (kind, ref) identity.
    await w.findAll('[data-testid="collection-item-remove"]')[0].trigger('click')
    expect(remove).toHaveBeenCalledWith('col_1', 'highlight', 'h1')
  })

  it('deletes a collection', async () => {
    const del = vi.spyOn(api, 'deleteCollection').mockResolvedValue([])
    const w = mountView()
    await flushPromises()
    await w.find('[aria-label="Delete collection"]').trigger('click')
    await flushPromises()
    expect(del).toHaveBeenCalledWith('col_1')
    expect(w.text()).toContain('No collections yet')
  })
})
