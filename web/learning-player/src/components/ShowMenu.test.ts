import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import type { Podcast } from '../services/types'
import { useAuthStore } from '../stores/auth'
import { useLibraryStore } from '../stores/library'
import ShowMenu from './ShowMenu.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/login', name: 'login', component: { template: '<div/>' } },
  ],
})
const SHOW: Podcast = {
  feed_id: 'p01',
  title: 'Singletrack Sessions',
  artwork_url: null,
  image_url: null,
  description: null,
  episode_count: 4,
}

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.restoreAllMocks())

async function open() {
  const w = mount(ShowMenu, {
    props: { show: SHOW },
    global: { plugins: [i18n, router], stubs: { teleport: true } },
  })
  await w.get('[data-testid="overflow-trigger"]').trigger('click')
  await flushPromises()
  return w
}

describe('ShowMenu', () => {
  it('holds every show action — follow, save, add to board — as menu rows', async () => {
    useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    const w = await open()
    const items = w.get('[data-testid="overflow-menu"]').findAll('[role="menuitem"]')
    expect(items).toHaveLength(3)
    expect(items[0].text()).toContain('Follow')
    expect(w.find('[data-testid="favorite-button"]').exists()).toBe(true)
  })

  it('follows the show from the menu (signed in)', async () => {
    useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    const toggle = vi.spyOn(useLibraryStore(), 'toggle').mockResolvedValue()
    const w = await open()
    await w.get('[data-testid="follow-show"]').trigger('click')
    await flushPromises()
    expect(toggle).toHaveBeenCalledWith('p01', { title: 'Singletrack Sessions' })
  })

  it('routes a signed-out Follow to sign-in instead of calling the API (#1590)', async () => {
    const follow = vi.spyOn(api, 'followShow')
    const w = await open()
    expect(w.get('[data-testid="follow-show"]').text()).toContain('Sign in to follow')
    await w.get('[data-testid="follow-show"]').trigger('click')
    await flushPromises()
    expect(follow).not.toHaveBeenCalled()
  })
})
