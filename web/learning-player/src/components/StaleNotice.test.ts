import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { setForcedOffline } from '../composables/useOnline'
import en from '../i18n/locales/en.json'
import StaleNotice from './StaleNotice.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
afterEach(() => {
  setForcedOffline(false)
  window.dispatchEvent(new Event('online'))
})

describe('StaleNotice', () => {
  it('says nothing in forced offline — the app-wide strip already does (operator 2026-10-09)', async () => {
    setForcedOffline(true)
    const w = mount(StaleNotice, { global: { plugins: [i18n] } })
    await w.vm.$nextTick()
    expect(w.find('[data-testid="stale-notice"]').exists()).toBe(false)
  })

  it('stays for a real outage, with the retry the page has no other copy of', async () => {
    window.dispatchEvent(new Event('offline'))
    const w = mount(StaleNotice, { global: { plugins: [i18n] } })
    await w.vm.$nextTick()
    expect(w.get('[data-testid="stale-notice"]').text()).toContain("You're offline")
    await w.get('[data-testid="stale-retry"]').trigger('click')
    expect(w.emitted('retry')).toHaveLength(1)
  })
})
