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
    expect(w.text()).toContain('2 highlights')
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

  it('opens a collection and renders its highlights', async () => {
    const detail: CollectionDetail = {
      collection: col(),
      highlights: [
        {
          id: 'h1', episode_slug: 'ep', kind: 'span', start_ms: 60_000, end_ms: null,
          char_start: null, char_end: null, segment_ids: [], quote_text: 'a line',
          speaker: null, source_insight_id: null, color: null, created_at: 1,
          anchor_status: null, graph_refs: [{ id: 'topic:ai', kind: 'topic', label: 'AI' }],
        },
      ],
    }
    vi.spyOn(api, 'getCollection').mockResolvedValue(detail)
    const w = mountView()
    await flushPromises()
    await w.findAll('button').find((b) => b.text().includes('AI takes'))!.trigger('click')
    await flushPromises()
    expect(w.text()).toContain('a line')
    expect(w.text()).toContain('AI')
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
