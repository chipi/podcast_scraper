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
  // The store sends ITEM-level intents now (#1925); echo a plausible server answer.
  vi.spyOn(api, 'addQueueItem').mockImplementation(async (slug) => [slug])
  vi.spyOn(api, 'removeQueueItem').mockResolvedValue([])
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

/**
 * A stale queue is a cached copy that was never revalidated, and every mutation refuses from it —
 * a PUT sends the whole list and would delete the server's queue. The control used to stay enabled
 * and silently do nothing, which is what a dead button looks like (#1925 review).
 */
describe('QueueButton offline', () => {
  it('still works when the queue is stale — the tap goes to the outbox', async () => {
    // It used to be DISABLED here, because every mutation went through a whole-list PUT that
    // would have deleted the server's queue. Add and remove are item-level and idempotent now, so
    // an offline tap is recorded and replayed; only reordering still needs a live list (#1925).
    useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    const queue = useQueueStore()
    queue.items = []
    queue.loaded = true
    queue.stale = true

    const w = mountBtn()
    await flushPromises()
    const btn = w.find('button')
    expect(btn.attributes('disabled')).toBeUndefined()

    await btn.trigger('click')
    await flushPromises()
    expect(api.addQueueItem).toHaveBeenCalledWith('ep-1')
  })
})
