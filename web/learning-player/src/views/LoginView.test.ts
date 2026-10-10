import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import * as api from '../services/api'

const tierSwitchEnabled = vi.fn(() => false)
vi.mock('../services/tier', async (orig) => ({
  ...(await orig<typeof import('../services/tier')>()),
  tierSwitchEnabled: () => tierSwitchEnabled(),
}))

const LoginView = (await import('./LoginView.vue')).default
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

async function mountLogin() {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/login', name: 'login', component: LoginView },
      { path: '/', name: 'home', component: { template: '<div/>' } },
      { path: '/about/:page', name: 'about-page', component: { template: '<div/>' } },
    ],
  })
  router.push('/login')
  await router.isReady()
  const w = mount(LoginView, { global: { plugins: [i18n, router] } })
  await flushPromises()
  return w
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.spyOn(api, 'getDevUsers').mockResolvedValue({ enabled: false, users: [] })
  vi.spyOn(api, 'getAuthProviders').mockResolvedValue([])
})
afterEach(() => vi.restoreAllMocks())

describe('LoginView: the DEV/PROD switch (2026-10-10)', () => {
  it('sits on the sign-in screen in the native internal app, which starts on PROD with no dev login', async () => {
    tierSwitchEnabled.mockReturnValue(true)
    const w = await mountLogin()
    expect(w.find('[data-testid="login-tier-switch"]').exists()).toBe(true)
  })

  it('is absent on the web and in a release build: no empty row above the title', async () => {
    tierSwitchEnabled.mockReturnValue(false)
    const w = await mountLogin()
    expect(w.find('[data-testid="login-tier-switch"]').exists()).toBe(false)
  })
})
