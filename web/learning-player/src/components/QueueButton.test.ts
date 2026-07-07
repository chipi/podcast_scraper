import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'
import { useQueueStore } from '../stores/queue'
import QueueButton from './QueueButton.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountBtn = () =>
  mount(QueueButton, { props: { slug: 'ep-1' }, global: { plugins: [i18n] } })

beforeEach(() => {
  setActivePinia(createPinia())
  vi.spyOn(api, 'putQueue').mockResolvedValue()
})
afterEach(() => vi.restoreAllMocks())

describe('QueueButton', () => {
  it('is hidden when signed out (auth-gated)', () => {
    expect(mountBtn().find('button').exists()).toBe(false)
  })

  it('signed in: toggles the queue and reflects state via aria-pressed', async () => {
    useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    const queue = useQueueStore()
    const w = mountBtn()
    const btn = w.find('button')
    expect(btn.exists()).toBe(true)
    expect(btn.attributes('aria-pressed')).toBe('false')
    expect(btn.attributes('aria-label')).toBe('Add to queue')

    await btn.trigger('click')
    expect(queue.has('ep-1')).toBe(true)
    expect(w.find('button').attributes('aria-pressed')).toBe('true')
    expect(w.find('button').attributes('aria-label')).toBe('Remove from queue')
  })
})
