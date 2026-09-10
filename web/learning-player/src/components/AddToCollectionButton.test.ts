import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import * as outbox from '../services/outbox'
import en from '../i18n/locales/en.json'
import type { Collection } from '../services/types'
import { useAuthStore } from '../stores/auth'
import AddToCollectionButton from './AddToCollectionButton.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: stub },
    { path: '/login', name: 'login', component: stub },
  ],
})

function col(over: Partial<Collection> = {}): Collection {
  return { id: 'col_1', name: 'Research', created_at: 1, count: 0, ...over }
}

async function mountIt(signedIn = true) {
  setActivePinia(createPinia())
  await router.push('/')
  await router.isReady()
  const w = mount(AddToCollectionButton, {
    props: { item: { kind: 'episode', ref: 'ep-x' } },
    global: { plugins: [i18n, router] },
  })
  if (signedIn) {
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    auth.loaded = true
  } else {
    useAuthStore().loaded = true
  }
  await flushPromises()
  return w
}

afterEach(() => vi.restoreAllMocks())

describe('AddToCollectionButton (#1839)', () => {
  it('opens the menu and pins the item to a chosen collection', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    const add = vi.spyOn(api, 'addToCollection').mockResolvedValue(col({ count: 1 }))
    const w = await mountIt()
    await w.get('[data-testid="add-to-collection"]').trigger('click')
    await flushPromises()
    expect(w.get('[data-testid="add-to-collection-menu"]').text()).toContain('Research')
    await w.get('[data-testid="add-to-collection-pick"]').trigger('click')
    expect(add).toHaveBeenCalledWith('col_1', { kind: 'episode', ref: 'ep-x' })
  })

  it('creates a new collection and pins into it', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([])
    const create = vi.spyOn(api, 'createCollection').mockResolvedValue(col({ id: 'col_2', name: 'New' }))
    const add = vi.spyOn(api, 'addToCollection').mockResolvedValue(col({ id: 'col_2', count: 1 }))
    const w = await mountIt()
    await w.get('[data-testid="add-to-collection"]').trigger('click')
    await flushPromises()
    await w.find('input[type="text"]').setValue('New')
    await w.find('form').trigger('submit')
    await flushPromises()
    expect(create).toHaveBeenCalledWith('New')
    expect(add).toHaveBeenCalledWith('col_2', { kind: 'episode', ref: 'ep-x' })
  })

  it('signed out: routes to sign-in instead of opening', async () => {
    const push = vi.spyOn(router, 'push')
    const w = await mountIt(false)
    await w.get('[data-testid="add-to-collection"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="add-to-collection-menu"]').exists()).toBe(false)
    expect(push).toHaveBeenCalledWith(expect.objectContaining({ name: 'login' }))
  })
})

describe('failures are visible, not swallowed (#2004 item 13)', () => {
  it('says so when the collection list cannot be loaded', async () => {
    // Was `.catch(() => [])`: a failed load rendered as "you have no collections", which is how a
    // user comes to believe their collections were never saved.
    vi.spyOn(api, 'getCollections').mockRejectedValue(new api.ApiError(401, 'nope'))
    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    expect(w.get('[data-testid="collection-error"]').text()).toBe(en.collections.loadFailed)
  })

  it('retries the load on the next open instead of latching an empty list', async () => {
    // `loaded` must not latch on failure, or the panel shows an empty list forever in a session.
    const spy = vi.spyOn(api, 'getCollections').mockRejectedValue(new api.ApiError(500, 'boom'))
    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    await w.get('button').trigger('click') // close
    spy.mockResolvedValue([col()])
    await w.get('button').trigger('click') // re-open
    await flushPromises()
    expect(spy).toHaveBeenCalledTimes(2)
    expect(w.find('[data-testid="collection-error"]').exists()).toBe(false)
    expect(w.text()).toContain('Research')
  })

  it('says so when adding to a collection fails, and claims nothing', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    // 422 = a REFUSAL. A 500 is transient and is now queued for replay, not surfaced.
    vi.spyOn(api, 'addToCollection').mockRejectedValue(new api.ApiError(422, 'refused'))
    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    await w.findAll('button').filter((b) => b.text().includes('Research'))[0].trigger('click')
    await flushPromises()
    expect(w.get('[data-testid="collection-error"]').text()).toBe(en.collections.addFailed)
    expect(w.text()).not.toContain('✓')
  })

  it('says so when creating fails, and KEEPS the typed name', async () => {
    // Retyping a name the app lost is the app's mistake charged to the user.
    vi.spyOn(api, 'getCollections').mockResolvedValue([])
    const create = vi.spyOn(api, 'createCollection').mockRejectedValue(new api.ApiError(422, 'refused'))
    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    const input = w.get('input')
    await input.setValue('Tech')
    await w.get('form').trigger('submit')
    await flushPromises()
    expect(create).toHaveBeenCalledWith('Tech')
    expect(w.get('[data-testid="collection-error"]').text()).toBe(en.collections.createFailed)
    expect((input.element as HTMLInputElement).value).toBe('Tech')
  })
})

describe('menu a11y (#2004 #9)', () => {
  it('exposes aria-expanded and closes on Escape / outside-click', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    const w = await mountIt()
    const trigger = w.get('[data-testid="add-to-collection"]')
    expect(trigger.attributes('aria-haspopup')).toBe('true')
    expect(trigger.attributes('aria-expanded')).toBe('false')

    await trigger.trigger('click')
    await flushPromises()
    expect(trigger.attributes('aria-expanded')).toBe('true')
    expect(w.find('[data-testid="add-to-collection-menu"]').exists()).toBe(true)

    // Escape closes.
    await w.get('div.relative').trigger('keydown', { key: 'Escape' })
    expect(w.find('[data-testid="add-to-collection-menu"]').exists()).toBe(false)
    expect(trigger.attributes('aria-expanded')).toBe('false')

    // Reopen, then an outside pointerdown closes.
    await trigger.trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="add-to-collection-menu"]').exists()).toBe(true)
    document.dispatchEvent(new Event('pointerdown'))
    await flushPromises()
    expect(w.find('[data-testid="add-to-collection-menu"]').exists()).toBe(false)
  })
})

describe('a transient failure is QUEUED, not lost (#2004 item 13 — root cause)', () => {
  it('queues a create that failed transiently, and shows it immediately', async () => {
    // Collections was the only per-user write with no outbox: favourites, queue, highlights, notes
    // and follows all replay. That asymmetry is why a flaky moment was invisible everywhere else and
    // permanent here — not a collections-specific bug.
    const enqueued: unknown[] = []
    vi.spyOn(outbox, 'enqueue').mockImplementation((a) => void enqueued.push(a))
    vi.spyOn(api, 'getCollections').mockResolvedValue([])
    vi.spyOn(api, 'createCollection').mockRejectedValue(new api.ApiError(503, 'gateway'))

    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    await w.get('input').setValue('Tech')
    await w.get('form').trigger('submit')
    await flushPromises()

    // The offline "create AND add" queues BOTH halves (#2004 #5): the create with a client-minted id,
    // and the pin targeting that same id — the server honours the id on replay so the pin lands.
    // Queuing only the create (as before) silently dropped the item the user was adding.
    expect(enqueued).toHaveLength(2)
    expect(enqueued[0]).toMatchObject({ op: 'collection.create', name: 'Tech' })
    const clientId = (enqueued[0] as { clientId: string }).clientId
    expect(clientId).toMatch(/^col_[a-z0-9]+$/)
    expect(enqueued[1]).toMatchObject({ op: 'collection.addItem', collectionId: clientId })
    expect(w.text()).toContain('Tech')
    expect(w.find('[data-testid="collection-error"]').exists()).toBe(false)
  })

  it('queues an add that failed transiently', async () => {
    const enqueued: unknown[] = []
    vi.spyOn(outbox, 'enqueue').mockImplementation((a) => void enqueued.push(a))
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    vi.spyOn(api, 'addToCollection').mockRejectedValue(new api.ApiError(503, 'gateway'))

    const w = await mountIt()
    await w.get('button').trigger('click')
    await flushPromises()
    await w.findAll('button').filter((b) => b.text().includes('Research'))[0].trigger('click')
    await flushPromises()

    expect(enqueued[0]).toMatchObject({ op: 'collection.addItem', collectionId: 'col_1' })
    expect(w.find('[data-testid="collection-error"]').exists()).toBe(false)
  })
})
