import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { Collection } from '../services/types'
import { useAuthStore } from '../stores/auth'
import AddToCollectionButton from './AddToCollectionButton.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: stub },
    { path: '/login', name: 'login', component: stub },
  ],
})

function col(over: Partial<Collection> = {}): Collection {
  return { id: 'col_1', name: 'Research', created_at: 1, count: 0, ...over }
}

async function mountIt(signedIn = true) {
  setActivePinia(createPinia())
  await router.push('/')
  await router.isReady()
  const w = mount(AddToCollectionButton, {
    props: { item: { kind: 'episode', ref: 'ep-x' } },
    global: { plugins: [i18n, router] },
  })
  if (signedIn) {
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    auth.loaded = true
  } else {
    useAuthStore().loaded = true
  }
  await flushPromises()
  return w
}

afterEach(() => vi.restoreAllMocks())

describe('AddToCollectionButton (#1839)', () => {
  it('opens the menu and pins the item to a chosen collection', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    const add = vi.spyOn(api, 'addToCollection').mockResolvedValue(col({ count: 1 }))
    const w = await mountIt()
    await w.get('[data-testid="add-to-collection"]').trigger('click')
    await flushPromises()
    expect(w.get('[data-testid="add-to-collection-menu"]').text()).toContain('Research')
    await w.get('[data-testid="add-to-collection-pick"]').trigger('click')
    expect(add).toHaveBeenCalledWith('col_1', { kind: 'episode', ref: 'ep-x' })
  })

  it('creates a new collection and pins into it', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([])
    const create = vi.spyOn(api, 'createCollection').mockResolvedValue(col({ id: 'col_2', name: 'New' }))
    const add = vi.spyOn(api, 'addToCollection').mockResolvedValue(col({ id: 'col_2', count: 1 }))
    const w = await mountIt()
    await w.get('[data-testid="add-to-collection"]').trigger('click')
    await flushPromises()
    await w.find('input[type="text"]').setValue('New')
    await w.find('form').trigger('submit')
    await flushPromises()
    expect(create).toHaveBeenCalledWith('New')
    expect(add).toHaveBeenCalledWith('col_2', { kind: 'episode', ref: 'ep-x' })
  })

  it('signed out: routes to sign-in instead of opening', async () => {
    const push = vi.spyOn(router, 'push')
    const w = await mountIt(false)
    await w.get('[data-testid="add-to-collection"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="add-to-collection-menu"]').exists()).toBe(false)
    expect(push).toHaveBeenCalledWith(expect.objectContaining({ name: 'login' }))
  })
})
