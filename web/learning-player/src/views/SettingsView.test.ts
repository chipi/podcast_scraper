import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
// DeviceSettings renders nothing off-native — on the web there is no offline audio to configure —
// so without this the placement assertions below would pass or fail for the wrong reason.
vi.mock('../services/native', () => ({ isNative: () => true }))
vi.mock('../services/downloadScheduler', () => ({
  DEFAULT_POLICY: 'wifi-only',
  applyDownloadCap: async () => {},
  getNetworkPolicy: async () => 'wifi-only',
  setNetworkPolicy: async () => {},
}))
vi.mock('../services/deviceStore', () => ({
  getDeviceJson: async () => null,
  setDeviceJson: async () => {},
}))

const SettingsView = (await import('./SettingsView.vue')).default

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/settings', name: 'settings', component: SettingsView },
    { path: '/profile', name: 'profile', component: stub },
  ],
})

async function mountView() {
  await router.push({ name: 'settings' })
  await router.isReady()
  return mount(SettingsView, { global: { plugins: [i18n, router] } })
}

beforeEach(() => {
  // This view reads the downloads store (#1905); without an active Pinia the mount throws.
  setActivePinia(createPinia())
})

describe('SettingsView (#8)', () => {
  it('carries the Device section — this page is how the app is set up', async () => {
    // It lived at the bottom of the profile, whose subject is who you are. Its own contract is
    // "settings that belong to THIS PHONE rather than to the account", shared by every account
    // that signs in on the handset — which is this page's subject, not the profile's.
    const w = await mountView()
    expect(w.find('[data-testid="device-settings"]').exists(), 'Device is not on Settings').toBe(true)
  })

  it('leads with what you can CHANGE, not the version you can only read', async () => {
    const w = await mountView()
    const html = w.html()
    expect(html.indexOf('device-settings')).toBeGreaterThan(-1)
    expect(
      html.indexOf('device-settings') < html.indexOf('settings-version'),
      'the About block comes before the settings you can actually change',
    ).toBe(true)
  })

  it('surfaces the build identity and a help entry', async () => {
    const w = await mountView()
    expect(w.find('[data-testid="settings-view"]').exists()).toBe(true)
    // __APP_VERSION__ comes from the shared vite define (package.json version).
    expect(w.get('[data-testid="settings-version"]').text()).toMatch(/^v\d+\.\d+\.\d+$/)
    expect(w.get('[data-testid="settings-copy"]').text()).toContain('Copy build info')
    expect(w.get('[data-testid="settings-help"]').exists()).toBe(true)
    expect(w.text()).toContain('web') // platform; capitalize is CSS-only, DOM text stays 'web'
  })

  it('links back to Profile', async () => {
    const w = await mountView()
    expect(w.find('a[href="/profile"]').exists()).toBe(true)
  })
})
