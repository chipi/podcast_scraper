import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import OfflineDownloadsView from './OfflineDownloadsView.vue'
import { useAuthStore } from '../stores/auth'
import { useDownloadsStore } from '../stores/downloads'

/**
 * "On this device" — the offline, signed-out fallback (operator 2026-09-23).
 *
 * The behaviour worth pinning is not that it renders a list. It is that it renders ONLY in the one
 * state it exists for. Downloads are namespaced per account so a shared phone cannot show one
 * person's listening history to the next, and this route deliberately reads the last account's
 * registry — so its gates ARE the privacy boundary, not a nicety.
 */
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: stub },
    { path: '/welcome', name: 'landing', component: stub },
    { path: '/offline', name: 'offline-downloads', component: stub },
    { path: '/episode/:slug', name: 'player', component: stub },
  ],
})

let online = false
vi.mock('../composables/useOnline', () => ({
  useOnline: () => ({ isOnline: { value: online } }),
}))

async function mountIt() {
  setActivePinia(createPinia())
  await router.push('/offline')
  await router.isReady()
  const w = mount(OfflineDownloadsView, { global: { plugins: [i18n, router] } })
  await flushPromises()
  return w
}

beforeEach(() => {
  online = false
  vi.restoreAllMocks()
})

describe('OfflineDownloadsView gates', () => {
  it('redirects a SIGNED-IN user home — the full Library is already there', async () => {
    setActivePinia(createPinia())
    const replace = vi.spyOn(router, 'replace').mockResolvedValue(undefined)
    await router.push('/offline')
    const auth = useAuthStore()
    auth.user = { user_id: 'u1' } as never
    mount(OfflineDownloadsView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(replace).toHaveBeenCalledWith({ name: 'home' })
  })

  it('redirects an ONLINE visitor to the landing — where they can actually sign in', async () => {
    // The gate that keeps this from becoming a back-door into another account's history while the
    // network is up. Offline is the whole justification; online it has none.
    online = true
    setActivePinia(createPinia())
    const replace = vi.spyOn(router, 'replace').mockResolvedValue(undefined)
    await router.push('/offline')
    mount(OfflineDownloadsView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(replace).toHaveBeenCalledWith({ name: 'landing' })
  })

  it('offline and signed out, adopts the last account and says so when there is nothing', async () => {
    const w = await mountIt()
    expect(w.find('[data-testid="offline-downloads"]').exists()).toBe(true)
    // No registry on this device: an explicit line, not an empty page that reads as broken.
    expect(w.find('[data-testid="offline-downloads-empty"]').exists()).toBe(true)
  })

  it('never writes: adoption is a read', async () => {
    setActivePinia(createPinia())
    const downloads = useDownloadsStore()
    const persist = vi.spyOn(downloads, '_persist')
    await router.push('/offline')
    mount(OfflineDownloadsView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(persist).not.toHaveBeenCalled()
  })
})
