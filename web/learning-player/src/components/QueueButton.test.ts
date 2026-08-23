import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'
import { useQueueStore } from '../stores/queue'
import QueueButton from './QueueButton.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountBtn = () => mount(QueueButton, { props: { slug: 'ep-1' }, global: { plugins: [i18n] } })

beforeEach(() => {
  setActivePinia(createPinia())
  // toggle()/add() call ensureLoaded() → getQueue() before mutating (RFC-099 §4,
  // prevents a late load clobbering the optimistic add). Mock it too, else the real
  // fetch rejects and toggle() bails before the add.
  vi.spyOn(api, 'getQueue').mockResolvedValue([])
  vi.spyOn(api, 'putQueue').mockResolvedValue()
})
afterEach(() => vi.restoreAllMocks())

describe('QueueButton', () => {
  it('renders signed out as a sign-in teaser, not hidden (#1590)', async () => {
    // It used to be v-if="auth.isAuthenticated". A signed-out listener therefore saw no evidence
    // the queue existed — the capability was invisible to exactly the people deciding whether to
    // sign up. It renders now; the tap explains the requirement.
    const w = mountBtn()
    const btn = w.find('button')
    expect(btn.exists()).toBe(true)
    expect(btn.attributes('aria-label')).toContain('Sign in')
    // No aria-pressed when signed out: there is no per-user state to report yet.
    expect(btn.attributes('aria-pressed')).toBeUndefined()
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
    // toggle() is fire-and-forget from the click handler and its optimistic push now
    // sits behind ensureLoaded()→getQueue(); flush the microtask chain before asserting.
    await flushPromises()
    expect(queue.has('ep-1')).toBe(true)
    expect(w.find('button').attributes('aria-pressed')).toBe('true')
    expect(w.find('button').attributes('aria-label')).toBe('Remove from queue')
  })
})
