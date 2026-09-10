import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import OfflineBanner from './OfflineBanner.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
afterEach(() => window.dispatchEvent(new Event('online')))

describe('OfflineBanner', () => {
  it('renders nothing while online', () => {
    window.dispatchEvent(new Event('online'))
    const w = mount(OfflineBanner, { global: { plugins: [i18n] } })
    expect(w.find('[data-testid="offline-banner"]').exists()).toBe(false)
  })

  it('shows the saved-content notice when the device goes offline', async () => {
    const w = mount(OfflineBanner, { global: { plugins: [i18n] } })
    window.dispatchEvent(new Event('offline'))
    await w.vm.$nextTick()
    const bar = w.find('[data-testid="offline-banner"]')
    expect(bar.exists()).toBe(true)
    expect(bar.text()).toContain('Offline')
    expect(bar.attributes('role')).toBe('status')
  })
})
