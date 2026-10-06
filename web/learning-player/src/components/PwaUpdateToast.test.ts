import { describe, expect, it, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import { ref } from 'vue'
import { createPinia, setActivePinia } from 'pinia'
import en from '../i18n/locales/en.json'
import { usePlayerStore } from '../stores/player'

// Mock the composable so we can drive needRefresh + the click handlers directly.
const needRefresh = ref(false)
const applyUpdate = vi.fn().mockResolvedValue(undefined)
const dismissUpdate = vi.fn()

vi.mock('../composables/usePwaUpdate', () => ({
  usePwaUpdate: () => ({
    needRefresh,
    offlineReady: ref(false),
    applyUpdate,
    dismissUpdate,
  }),
}))

const native = vi.hoisted(() => ({ value: false }))
vi.mock('../services/native', async (orig) => ({
  ...(await orig<typeof import('../services/native')>()),
  isNative: () => native.value,
}))

import PwaUpdateToast from './PwaUpdateToast.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function mountToast() {
  return mount(PwaUpdateToast, { global: { plugins: [i18n] } })
}

describe('PwaUpdateToast', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    native.value = false
    needRefresh.value = false
    applyUpdate.mockClear()
    dismissUpdate.mockClear()
  })

  it('renders nothing when needRefresh is false', () => {
    const wrapper = mountToast()
    expect(wrapper.find('[data-testid="pwa-update-toast"]').exists()).toBe(false)
  })

  it('renders the toast when needRefresh is true', async () => {
    needRefresh.value = true
    const wrapper = mountToast()
    const toast = wrapper.find('[data-testid="pwa-update-toast"]')
    expect(toast.exists()).toBe(true)
    // English copy assertions (guards accidental i18n key regression).
    expect(toast.text()).toContain('New version available')
    expect(toast.text()).toContain('Reload')
    expect(toast.text()).toContain('Later')
  })

  it('reload button calls applyUpdate()', async () => {
    needRefresh.value = true
    const wrapper = mountToast()
    await wrapper.find('[data-testid="pwa-update-reload"]').trigger('click')
    expect(applyUpdate).toHaveBeenCalledTimes(1)
  })

  it('dismiss button calls dismissUpdate()', async () => {
    needRefresh.value = true
    const wrapper = mountToast()
    await wrapper.find('[data-testid="pwa-update-dismiss"]').trigger('click')
    expect(dismissUpdate).toHaveBeenCalledTimes(1)
  })

  it('is announced to assistive tech via role=status + aria-live=polite', async () => {
    needRefresh.value = true
    const wrapper = mountToast()
    const toast = wrapper.find('[data-testid="pwa-update-toast"]')
    expect(toast.attributes('role')).toBe('status')
    expect(toast.attributes('aria-live')).toBe('polite')
  })

  // A fresh app update leaves a new service worker waiting, so the toast appeared on the FIRST
  // launch after every install, pinned bottom-right over the tab bar: a tap on Library landed on
  // the toast and did nothing (Android AppJourneyTests#test05, 2026-10-05). Native updates arrive
  // with the store build and have their own banner; the native app registers no worker at all.
  it('never renders in the native app', () => {
    native.value = true
    needRefresh.value = true
    const wrapper = mountToast()
    expect(wrapper.find('[data-testid="pwa-update-toast"]').exists()).toBe(false)
  })

  it('sits above the tab bar on a phone, and above the mini-player too when one is loaded', async () => {
    needRefresh.value = true
    const wrapper = mountToast()
    const cls = () => wrapper.find('[data-testid="pwa-update-toast"]').classes()
    expect(cls()).toContain('bottom-[calc(4.25rem+env(safe-area-inset-bottom))]')
    expect(cls()).toContain('sm:bottom-4')
    usePlayerStore().currentSlug = 'some-episode'
    await wrapper.vm.$nextTick()
    expect(cls()).toContain('bottom-[calc(8.25rem+env(safe-area-inset-bottom))]')
    expect(cls()).toContain('sm:bottom-[calc(6rem+env(safe-area-inset-bottom))]')
  })
})
