import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import { useOnline } from '../composables/useOnline'
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
    { path: '/about/:page', name: 'about-page', component: stub, props: true },
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
    expect(w.find('[data-testid="settings-help"]').exists()).toBe(true)
    expect(w.text()).toContain('web') // platform; capitalize is CSS-only, DOM text stays 'web'
  })

  it('links back to Profile', async () => {
    const w = await mountView()
    expect(w.find('a[href="/profile"]').exists()).toBe(true)
  })

  it('Config: the offline-mode toggle drives forced-offline; reclaim + volume controls render', async () => {
    const { forcedOffline, setForcedOffline } = useOnline()
    setForcedOffline(false)
    const w = await mountView()

    const toggle = w.find('[data-testid="settings-offline-mode"]')
    expect(toggle.exists()).toBe(true)
    expect((toggle.element as HTMLInputElement).checked).toBe(false)
    await toggle.setValue(true) // change → setForcedOffline(true)
    expect(forcedOffline.value).toBe(true)
    setForcedOffline(false) // cleanup the singleton for other tests

    // Reclaim actions (native mocked true → clear-downloads shows) + the Playback volume control.
    expect(w.find('[data-testid="settings-clear-cache"]').exists()).toBe(true)
    expect(w.find('[data-testid="settings-clear-downloads"]').exists()).toBe(true)
    expect(w.text()).toContain('Volume')
  })

  it('About & legal: Support link + the three placeholder pages route correctly', async () => {
    const w = await mountView()
    expect(w.find('[data-testid="settings-support"]').exists()).toBe(true)
    expect(w.find('[data-testid="settings-third-party"]').attributes('href')).toBe('/about/third-party')
    expect(w.find('[data-testid="settings-privacy"]').attributes('href')).toBe('/about/privacy')
    expect(w.find('[data-testid="settings-terms"]').attributes('href')).toBe('/about/terms')
  })
})

describe('Settings › Privacy (#2265)', () => {
  beforeEach(() => {
    try {
      localStorage.clear()
    } catch {
      /* storage may be unavailable */
    }
  })

  it('offers the usage-analytics toggle, ON by default', async () => {
    // The beta's closing-interview script tells every participant "analytics continues unless you
    // turn it off in Settings". Without this control that sentence is false, which is the whole
    // reason the toggle is in scope rather than deferred.
    const w = await mountView()
    const box = w.get('[data-testid="settings-share-analytics"]')
    expect((box.element as HTMLInputElement).checked).toBe(true)
    // Named on the input itself — the Android bridge reported such nodes with every name field
    // empty, so TalkBack announced a bare checkbox (#2156).
    expect(box.attributes('aria-label')).toBe(en.settings.shareAnalytics)
  })

  it('turning it off records the opt-out where Umami itself reads it', async () => {
    const w = await mountView()
    const box = w.get('[data-testid="settings-share-analytics"]')
    ;(box.element as HTMLInputElement).checked = false
    await box.trigger('change')
    // Umami's OWN flag, not a key of ours: its bundle checks `localStorage['umami.disabled']`, so
    // the tracker goes quiet even for a `track()` call that forgot our gate.
    expect(localStorage.getItem('umami.disabled')).toBe('1')
  })

  it('turning it back on clears the flag', async () => {
    localStorage.setItem('umami.disabled', '1')
    const w = await mountView()
    const box = w.get('[data-testid="settings-share-analytics"]')
    expect((box.element as HTMLInputElement).checked).toBe(false)
    ;(box.element as HTMLInputElement).checked = true
    await box.trigger('change')
    expect(localStorage.getItem('umami.disabled')).toBeNull()
  })

  it('hides the analytics id block when the account has none', async () => {
    // An account whose backfill has not run has no id; showing an empty code block would read as
    // a bug to the participant it is meant to help.
    const w = await mountView()
    expect(w.find('[data-testid="settings-analytics-id"]').exists()).toBe(false)
  })
})
