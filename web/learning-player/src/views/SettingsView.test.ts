import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import SettingsView from './SettingsView.vue'

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

describe('SettingsView (#8)', () => {
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
