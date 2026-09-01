import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import * as deviceStore from '../services/deviceStore'

const isNative = vi.fn(() => true)
const getNetworkPolicy = vi.fn()
const setNetworkPolicy = vi.fn()

vi.mock('../services/native', () => ({ isNative: () => isNative() }))
vi.mock('../services/downloadScheduler', () => ({
  DEFAULT_POLICY: 'wifi-only',
  getNetworkPolicy: () => getNetworkPolicy(),
  setNetworkPolicy: (p: string) => setNetworkPolicy(p),
}))
vi.mock('@capacitor/browser', () => ({ Browser: { open: vi.fn() } }))

const SettingsView = (await import('./SettingsView.vue')).default
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountView = () =>
  mount(SettingsView, {
    global: { plugins: [i18n], stubs: { RouterLink: { template: '<a><slot /></a>' } } },
  })

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

describe('SettingsView — downloads (#1905)', () => {
  it('is absent on the web, where a download control would promise what it cannot do', async () => {
    isNative.mockReturnValue(false)
    const w = mountView()
    await flushPromises()
    expect(w.find('[data-testid="settings-downloads"]').exists()).toBe(false)
  })

  it('defaults to Wi-Fi only', async () => {
    const w = mountView()
    await flushPromises()
    expect(w.find('[data-testid="settings-policy-wifi-only"]').attributes('aria-pressed')).toBe(
      'true',
    )
    expect(w.find('[data-testid="settings-policy-any"]').attributes('aria-pressed')).toBe('false')
  })

  it('reflects a stored preference for cellular', async () => {
    getNetworkPolicy.mockResolvedValue('any')
    const w = mountView()
    await flushPromises()
    expect(w.find('[data-testid="settings-policy-any"]').attributes('aria-pressed')).toBe('true')
  })

  it('persists the choice, which also releases anything waiting', async () => {
    const w = mountView()
    await flushPromises()
    await w.find('[data-testid="settings-policy-any"]').trigger('click')
    await flushPromises()
    expect(setNetworkPolicy).toHaveBeenCalledWith('any')
    expect(w.find('[data-testid="settings-policy-any"]').attributes('aria-pressed')).toBe('true')
  })

  it('reports storage used', async () => {
    const w = mountView()
    await flushPromises()
    expect(w.find('[data-testid="settings-storage"]').text()).toBe('0 MB')
  })
})
