import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { Collection, CollectionDetail } from '../services/types'
import { useAuthStore } from '../stores/auth'
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

const mountView = () => {
  setActivePinia(createPinia())
  return mount(CollectionsView, { global: { plugins: [i18n, router, createPinia()] } })
}

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

  it('hydrates episode titles, play-all queues them, and add-link pins a URL', async () => {
    const detail: CollectionDetail = {
      collection: col(),
      items: [
        { kind: 'episode', ref: 'ep-1' },
        { kind: 'episode', ref: 'ep-2' },
      ],
    }
    vi.spyOn(api, 'getCollection').mockResolvedValue(detail)
    vi.spyOn(api, 'getEpisode').mockResolvedValue({
      slug: 'ep-1', title: 'Ep One', podcast_title: 'Show',
    } as never)
    vi.spyOn(api, 'getQueue').mockResolvedValue([])
    vi.spyOn(api, 'putQueue').mockResolvedValue()
    const add = vi.spyOn(api, 'addToCollection').mockResolvedValue(col({ count: 3 }))
    const w = mountView()
    const auth = useAuthStore() // play-all is sign-in gated
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    auth.loaded = true
    await flushPromises()
    await w.findAll('button').find((b) => b.text().includes('AI takes'))!.trigger('click')
    await flushPromises()
    expect(w.text()).toContain('Ep One') // episode title hydrated from the slug

    const push = vi.spyOn(router, 'push')
    await w.get('[data-testid="collection-play-all"]').trigger('click')
    await flushPromises()
    expect(push).toHaveBeenCalledWith({ name: 'player', params: { slug: 'ep-1' } })

    const linkForm = w.findAll('form').find((f) => f.find('[data-testid="collection-add-link"]').exists())!
    await linkForm.find('[data-testid="collection-add-link"]').setValue('https://ex.com/a')
    await linkForm.trigger('submit')
    expect(add).toHaveBeenCalledWith('col_1', { kind: 'link', ref: 'https://ex.com/a' })
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

describe('a failed load is not an empty library (#2004 item 13)', () => {
  it('shows a retryable error instead of the "no collections yet" empty state', async () => {
    // The screen a user saw after creating a collection elsewhere: `getCollections().catch(() => [])`
    // plus a `loaded` latch turned any failure — including the 401 the API layer used to
    // manufacture — into "you have no collections yet".
    vi.spyOn(api, 'getCollections').mockRejectedValue(new api.ApiError(401, 'nope'))
    const w = mountView()
    await flushPromises()
    expect(w.find('[data-testid="collections-load-error"]').exists()).toBe(true)
    expect(w.text()).not.toContain(en.collections.empty)
  })

  it('still shows the real empty state when the account genuinely has none', async () => {
    // The other half: the fix must not turn "you have none" into an error.
    vi.spyOn(api, 'getCollections').mockResolvedValue([])
    const w = mountView()
    await flushPromises()
    expect(w.find('[data-testid="collections-load-error"]').exists()).toBe(false)
    expect(w.text()).toContain(en.collections.empty)
  })

  it('recovers on retry', async () => {
    const spy = vi.spyOn(api, 'getCollections').mockRejectedValue(new api.ApiError(500, 'boom'))
    const w = mountView()
    await flushPromises()
    spy.mockResolvedValue([col()])
    await w.get('[data-testid="section-retry"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="collections-load-error"]').exists()).toBe(false)
    expect(w.text()).toContain('AI takes')
  })
})
