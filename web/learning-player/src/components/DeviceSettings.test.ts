import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import * as deviceStore from '../services/deviceStore'
import { useDownloadsStore } from '../stores/downloads'

const isNative = vi.fn(() => true)
const getNetworkPolicy = vi.fn()
const setNetworkPolicy = vi.fn()

vi.mock('../services/native', () => ({ isNative: () => isNative() }))
vi.mock('../services/downloadScheduler', () => ({
  DEFAULT_POLICY: 'wifi-only',
  getNetworkPolicy: () => getNetworkPolicy(),
  setNetworkPolicy: (p: string) => setNetworkPolicy(p),
}))

const DeviceSettings = (await import('./DeviceSettings.vue')).default
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountIt = () => mount(DeviceSettings, { global: { plugins: [i18n] } })

beforeEach(() => {
  setActivePinia(createPinia())
  isNative.mockReturnValue(true)
  getNetworkPolicy.mockResolvedValue('wifi-only')
  setNetworkPolicy.mockResolvedValue(undefined)
  vi.spyOn(deviceStore, 'getDeviceJson').mockResolvedValue(null)
  vi.spyOn(deviceStore, 'setDeviceJson').mockResolvedValue()
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('DeviceSettings (#1905)', () => {
  it('is absent on the web, where there is no offline audio to configure', async () => {
    isNative.mockReturnValue(false)
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="device-settings"]').exists()).toBe(false)
  })

  it('says plainly that these are shared by everyone on the device', async () => {
    // They are NOT namespaced per account: whoever holds the phone decides how it uses their
    // data plan. Saying so is what stops it reading like a per-account preference.
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="device-settings"]').text()).toContain(en.profile.deviceHelp)
  })

  it('defaults to Wi-Fi only', async () => {
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="device-policy-wifi-only"]').attributes('aria-pressed')).toBe(
      'true',
    )
    expect(w.find('[data-testid="device-policy-any"]').attributes('aria-pressed')).toBe('false')
  })

  it('reflects a stored preference for cellular', async () => {
    getNetworkPolicy.mockResolvedValue('any')
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="device-policy-any"]').attributes('aria-pressed')).toBe('true')
  })

  it('persists the choice, which also releases anything waiting', async () => {
    const w = mountIt()
    await flushPromises()
    await w.find('[data-testid="device-policy-any"]').trigger('click')
    await flushPromises()
    expect(setNetworkPolicy).toHaveBeenCalledWith('any')
    expect(w.find('[data-testid="device-policy-any"]').attributes('aria-pressed')).toBe('true')
  })

  it('reports storage used for this device', async () => {
    const store = useDownloadsStore()
    store.loaded = true
    store.entries = {
      a: { slug: 'a', state: 'downloaded', updatedAt: 1, bytes: 20 * 1024 * 1024 },
    }
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="device-storage"]').text()).toBe('20 MB')
  })
})
